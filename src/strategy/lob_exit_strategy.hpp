/**
 * @file lob_exit_strategy.hpp
 * @brief Optimal exit via non-equilibrium statistical mechanics on a stochastic LOB field.
 *
 * Model:
 *   - Queue dynamics as metastable state decay (Kramers escape rate)
 *   - Self-exciting order flow via Hawkes process λ(t) = μ + R(t)
 *   - Entropy-producing liquidity shocks modelled through LOB field Shannon entropy
 *   - Optimal liquidation from minimising free-energy F = E_adv + γV_inv − T·S_lob + ε·C_lat
 *   - Exit surface from HJB value-function approximation V(q,λ) = A·q² + B|q|λ + Cλ²
 *   - Bayesian Kalman filter suppressing microstructure Brownian noise
 */

#pragma once

#include "user_strategy.hpp"
#include <array>
#include <cmath>
#include <numeric>
#include <unordered_map>

namespace hft {

// Divide raw fixed-point prices by this to bring them to ≈ 1.0 scale.
static constexpr double PRICE_NORM = 1e6;

// ============================================================================
// HawkesProcess
//
// Self-exciting point process for order-arrival clustering.
//   λ(t) = μ + R(t)
//   R updated recursively: R ← (R + α) · exp(−β·Δt)  on each event
//   Branching ratio ρ = α/β; subcritical ↔ ρ < 1
// ============================================================================
class HawkesProcess {
public:
    HawkesProcess(double mu, double alpha, double beta) noexcept
        : mu_(mu), alpha_(alpha), beta_(beta) {}

    void tick(double t_sec) noexcept {
        if (last_t_ > 0.0)
            R_ *= std::exp(-beta_ * (t_sec - last_t_));
        R_ += alpha_;
        last_t_ = t_sec;
    }

    [[nodiscard]] double intensity(double t_sec) const noexcept {
        if (last_t_ <= 0.0) return mu_;
        return mu_ + R_ * std::exp(-beta_ * (t_sec - last_t_));
    }

    [[nodiscard]] double baseline()        const noexcept { return mu_; }
    [[nodiscard]] double branching_ratio() const noexcept { return alpha_ / beta_; }

private:
    double mu_, alpha_, beta_;
    double R_      = 0.0;
    double last_t_ = 0.0;
};

// ============================================================================
// BayesianMicroFilter
//
// 2-state linear Kalman filter for microstructure noise suppression.
//   State:       x = [price, drift]
//   Process:     price(t+dt) = price(t) + drift(t)·dt  +  w_p
//                drift(t+dt) = drift(t)                +  w_d
//   Observation: z = price + v,   R_obs = (half_spread)²
//
// Suppresses transient Brownian fluctuation modes; outputs filtered mid-price
// and latent drift estimate.
// ============================================================================
class BayesianMicroFilter {
public:
    BayesianMicroFilter(double q_price, double q_drift) noexcept
        : qp_(q_price), qd_(q_drift) {}

    void update(double z, double spread, double dt) noexcept {
        if (!init_) {
            xp_ = z;
            xd_ = 0.0;
            P_[0][0] = 1.0;  P_[0][1] = 0.0;
            P_[1][0] = 0.0;  P_[1][1] = 0.01;
            init_ = true;
            return;
        }
        // Predict
        double xp_pred = xp_ + xd_ * dt;
        double xd_pred = xd_;
        double P00 = P_[0][0] + dt * (P_[0][1] + P_[1][0])
                   + dt * dt * P_[1][1] + qp_ * dt;
        double P01 = P_[0][1] + dt * P_[1][1];
        double P10 = P_[1][0] + dt * P_[1][1];
        double P11 = P_[1][1] + qd_ * dt;

        // Update — H = [1, 0]
        double half = spread * 0.5;
        double R    = half * half + 1e-30;
        double S    = P00 + R;
        double K0   = P00 / S;
        double K1   = P10 / S;
        double innov = z - xp_pred;

        xp_ = xp_pred + K0 * innov;
        xd_ = xd_pred + K1 * innov;
        P_[0][0] = std::max((1.0 - K0) * P00, 0.0);
        P_[0][1] = (1.0 - K0) * P01;
        P_[1][0] = P10 - K1 * P00;
        P_[1][1] = std::max(P11 - K1 * P01, 0.0);
    }

    [[nodiscard]] double filtered_price()    const noexcept { return xp_; }
    [[nodiscard]] double filtered_drift()    const noexcept { return xd_; }
    [[nodiscard]] double price_variance()    const noexcept { return P_[0][0]; }

private:
    double qp_, qd_;
    double xp_ = 0.0, xd_ = 0.0;
    double P_[2][2]{};
    bool   init_ = false;
};

// ============================================================================
// LOBFieldState
//
// Stochastic limit-order-field snapshot.
//   Depth profile:  φ(k) = size · exp(−κ·k)   over N_LEVELS price ticks
//                   κ = 0.5 / (1 + rel_spread_bps · 0.01)
//
//   Shannon entropy S = −Σ pₖ log pₖ   (0 → perfectly concentrated,
//                                        log N → perfectly uniform)
//
//   Kramers escape rate  k_esc = exp(−ΔV/D):
//     ΔV = (bid_size + ask_size) / D   (potential barrier ~ queue depth)
//     D  = spread_bps · hawkes_lam     (diffusion coefficient from order flow)
//     High k_esc → metastable; queue likely to deplete before fill.
//
//   Adverse-selection factor: buy-side imbalance when selling, vice versa.
// ============================================================================
class LOBFieldState {
public:
    static constexpr int N_LEVELS = 8;

    void update(double bid_sz, double ask_sz,
                double spread_bps, double hawkes_lam) noexcept {
        double kappa = 0.5 / (1.0 + spread_bps * 0.01);
        for (int k = 0; k < N_LEVELS; ++k) {
            double w    = std::exp(-kappa * static_cast<double>(k));
            bid_[k]     = bid_sz * w;
            ask_[k]     = ask_sz * w;
        }
        entropy_ = 0.5 * (level_entropy(bid_) + level_entropy(ask_));

        // Kramers escape rate (dimensionless probability in [0,1])
        double depth = bid_sz + ask_sz + 1.0;
        double D     = spread_bps * hawkes_lam + 1.0;
        kramers_     = std::exp(-(depth / D));
    }

    [[nodiscard]] double entropy()      const noexcept { return entropy_; }
    [[nodiscard]] double kramers_rate() const noexcept { return kramers_; }

    // Returns imbalance fraction [0, 1] that acts against our liquidation side.
    [[nodiscard]] double adverse_sel_factor(Side side) const noexcept {
        double b   = std::accumulate(bid_.begin(), bid_.end(), 0.0);
        double a   = std::accumulate(ask_.begin(), ask_.end(), 0.0);
        double imb = (b - a) / (b + a + 1e-10);
        return (side == Side::SELL) ? std::max(0.0,  imb)
                                    : std::max(0.0, -imb);
    }

private:
    static double level_entropy(const std::array<double, N_LEVELS>& d) noexcept {
        double tot = std::accumulate(d.begin(), d.end(), 0.0);
        if (tot <= 0.0) return 0.0;
        double S = 0.0;
        for (double v : d) {
            double p = v / tot;
            if (p > 1e-15) S -= p * std::log(p);
        }
        return S;
    }

    std::array<double, N_LEVELS> bid_{};
    std::array<double, N_LEVELS> ask_{};
    double entropy_  = 0.0;
    double kramers_  = 0.0;
};

// ============================================================================
// FreeEnergyFunctional
//
// Thermodynamic cost functional over normalised quantities ∈ [0, 1]:
//
//   F = c_as · adv_sel_01
//     + γ   · q_norm² · σ_norm²        (inventory potential)
//     − T   · S_norm                   (execution entropy bonus)
//     + ε   · lat_norm                 (latency dissipation)
//
// Parameters:
//   c_as    — adverse-selection weight
//   gamma   — inventory risk aversion
//   T_exec  — execution temperature (entropy weight)
//   eps_lat — latency dissipation coefficient
// ============================================================================
struct FreeEnergyFunctional {
    double c_as    = 0.30;
    double gamma   = 0.50;
    double T_exec  = 0.30;
    double eps_lat = 0.05;

    [[nodiscard]] double evaluate(double adv_sel_01,
                                   double q_norm,
                                   double sigma_norm,
                                   double S_norm,
                                   double lat_norm) const noexcept {
        return c_as    * adv_sel_01
             + gamma   * q_norm * q_norm * sigma_norm * sigma_norm
             - T_exec  * S_norm
             + eps_lat * lat_norm;
    }
};

// ============================================================================
// HJBExitSurface
//
// Hamilton–Jacobi–Bellman approximate value function for optimal liquidation.
//
//   V(q_norm, λ_norm) = A · q_norm²
//                     + B · |q_norm| · λ_norm
//                     + C · λ_norm²
//
// Coefficients are the closed-form solution to the backward Riccati ODE under
// linear-quadratic inventory dynamics with Hawkes-driven adverse selection.
//
// Exit condition: V(q, λ) + F > threshold
//
// Optimal instantaneous rate (gradient descent on V):
//   u*(q, λ, σ) = −∂V/∂q / (2A · σ²)
// ============================================================================
class HJBExitSurface {
public:
    struct Params {
        double A         = 0.30;
        double B         = 0.10;
        double C         = 0.05;
        double threshold = 0.25;
    };

    HJBExitSurface() noexcept : p_{} {}
    explicit HJBExitSurface(Params p) noexcept : p_(p) {}

    [[nodiscard]] double value(double q_norm, double lam_norm) const noexcept {
        return p_.A * q_norm  * q_norm
             + p_.B * std::abs(q_norm) * lam_norm
             + p_.C * lam_norm * lam_norm;
    }

    [[nodiscard]] bool triggers_exit(double q_norm,
                                      double lam_norm,
                                      double free_energy) const noexcept {
        return (value(q_norm, lam_norm) + free_energy) > p_.threshold;
    }

    // Gradient of V with respect to q_norm, for sizing the exit order.
    [[nodiscard]] double grad_q(double q_norm, double lam_norm) const noexcept {
        double sign_q = (q_norm > 0.0) ? 1.0 : ((q_norm < 0.0) ? -1.0 : 0.0);
        return 2.0 * p_.A * q_norm + p_.B * sign_q * lam_norm;
    }

private:
    Params p_;
};

// ============================================================================
// LOBExitStrategy
//
// Full strategy integrating:
//   • Hawkes self-excitation           (order-flow clustering)
//   • Bayesian Kalman microfilter      (noise / drift separation)
//   • Stochastic LOB field             (entropy, Kramers metastability)
//   • Free-energy functional           (cost of holding vs. exiting)
//   • HJB exit surface                 (Hamilton–Jacobi–Bellman decision)
// ============================================================================
class LOBExitStrategy : public UserStrategy {
public:
    LOBExitStrategy();

    void onInit()                              override;
    void onTick(const Tick& tick)              override;
    void onOrderResponse(const OrderResponse&) override;
    void onShutdown()                          override;
    const char* name() const                   override { return "LOBExitStrategy"; }

private:
    // Physics engine
    HawkesProcess        hawkes_{50.0, 1.5, 500.0};   // μ=50, α=1.5, β=500 → ρ=0.003
    BayesianMicroFilter  kfilter_{1e-6, 1e-8};         // process noise (normalised)
    LOBFieldState        lob_;
    FreeEnergyFunctional free_energy_;
    HJBExitSurface       hjb_;

    // Strategy state (all in normalised units)
    double inventory_     = 0.0;
    double sigma_norm_    = 0.01;
    double ema_sigma_     = 0.01;
    double last_free_e_   = 0.0;
    double last_t_sec_    = 0.0;
    uint64_t tick_count_  = 0;
    uint64_t orders_sent_ = 0;
    uint64_t exits_fired_ = 0;

    // Side tracking for fills
    std::unordered_map<uint64_t, Side> pending_;

    static constexpr uint64_t WARMUP_TICKS  = 50;
    static constexpr double   MAX_INV       = 100.0;
    static constexpr double   ENTRY_LOT     = 10.0;
    static constexpr double   EXIT_LOT      = 10.0;
    static constexpr double   SIGMA_ALPHA   = 0.05;   // EMA smoothing
    static constexpr double   SIGMA_REF_BPS = 10.0;   // reference vol (bps)
};

// Forward declaration — defined in lob_exit_strategy.cpp, linked into hftperf.
std::unique_ptr<UserStrategy> create_lob_exit_strategy();

} // namespace hft
