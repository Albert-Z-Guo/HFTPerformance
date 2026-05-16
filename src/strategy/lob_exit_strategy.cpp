/**
 * @file lob_exit_strategy.cpp
 * @brief LOBExitStrategy implementation — non-equilibrium exit model.
 *
 * Per-tick pipeline:
 *   1. Hawkes:       record order-flow event, compute self-exciting intensity λ(t)
 *   2. Bayesian:     Kalman update on mid-price; extract filtered drift + uncertainty
 *   3. LOB field:    update stochastic depth field, entropy S, Kramers escape rate k
 *   4. Free energy:  F = c_as·adv_sel + γ·q²·σ² − T·S_norm + ε·lat_norm
 *   5. HJB surface:  exit when V(q_norm, λ_norm) + F > threshold
 */

#include "strategy/lob_exit_strategy.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace hft {

LOBExitStrategy::LOBExitStrategy() = default;

void LOBExitStrategy::onInit() {
    std::cout << "[LOBExitStrategy] Initialised — non-equilibrium LOB exit model\n"
              << "  Hawkes:  μ=" << hawkes_.baseline()
              << "  ρ=" << hawkes_.branching_ratio() << " (subcritical)\n"
              << "  HJB threshold = " << HJBExitSurface::Params{}.threshold << "\n"
              << "  Max inventory = " << MAX_INV << "\n";
}

void LOBExitStrategy::onTick(const Tick& tick) {
    record_timestamp("lob_entry");
    ++tick_count_;

    // ── Time bookkeeping ──────────────────────────────────────────────────
    double t_sec = static_cast<double>(tick.timestamp) * 1e-9;
    double dt    = (last_t_sec_ > 0.0) ? (t_sec - last_t_sec_) : 1e-5;
    if (dt <= 0.0) dt = 1e-6;
    last_t_sec_ = t_sec;

    // ── Normalised price quantities ───────────────────────────────────────
    // Divide by PRICE_NORM so mid ≈ 1.0, spread ≈ 1e-4 (1 bp at typical sim prices)
    double bid_n    = static_cast<double>(tick.bid_price) / PRICE_NORM;
    double ask_n    = static_cast<double>(tick.ask_price) / PRICE_NORM;
    double mid_n    = 0.5 * (bid_n + ask_n);
    double spread_n = ask_n - bid_n;

    // Relative spread in basis points
    double spread_bps = (mid_n > 0.0) ? (spread_n / mid_n * 1e4) : 1.0;

    // ── Step 1: Hawkes process ────────────────────────────────────────────
    // Each market data tick is an order-flow event; intensity rises then decays.
    hawkes_.tick(t_sec);
    double lam      = hawkes_.intensity(t_sec);
    double lam_norm = lam / (hawkes_.baseline() + 1e-10);   // normalised: 1 = baseline
    record_timestamp("hawkes_done");

    // ── Step 2: Bayesian microstructure filter ────────────────────────────
    kfilter_.update(mid_n, spread_n, dt);
    double p_filt  = kfilter_.filtered_price();
    double d_filt  = kfilter_.filtered_drift();

    // Instantaneous vol in basis points, then EMA-smooth it.
    // σ ≈ |drift|·√dt (displacement over one step) + measurement uncertainty
    double sigma_raw = (std::abs(d_filt) * std::sqrt(dt)
                        + std::sqrt(std::max(kfilter_.price_variance(), 0.0)));
    double sigma_bps = (mid_n > 0.0) ? (sigma_raw / mid_n * 1e4) : 1.0;
    ema_sigma_ = SIGMA_ALPHA * sigma_bps + (1.0 - SIGMA_ALPHA) * ema_sigma_;
    sigma_norm_ = std::max(ema_sigma_, 1e-6) / SIGMA_REF_BPS;   // normalise by ref vol
    record_timestamp("bayes_done");

    // ── Step 3: LOB field ─────────────────────────────────────────────────
    lob_.update(static_cast<double>(tick.bid_size),
                static_cast<double>(tick.ask_size),
                spread_bps, lam);
    double S_lob  = lob_.entropy();
    double S_max  = std::log(static_cast<double>(LOBFieldState::N_LEVELS));
    double S_norm = S_lob / (S_max + 1e-10);       // normalised entropy ∈ [0,1]
    record_timestamp("lob_field_done");

    if (tick_count_ < WARMUP_TICKS) return;

    // ── Step 4: free-energy functional ───────────────────────────────────
    double q_norm    = inventory_ / MAX_INV;               // normalised inventory
    Side   exit_side = (inventory_ >= 0.0) ? Side::SELL : Side::BUY;
    double adv_sel   = lob_.adverse_sel_factor(exit_side); // [0,1]
    double lat_norm  = 0.5 + 0.5 * lob_.kramers_rate();   // map Kramers → lat proxy

    last_free_e_ = free_energy_.evaluate(adv_sel, q_norm,
                                          sigma_norm_, S_norm, lat_norm);
    record_timestamp("free_energy_done");

    // ── Entry logic ───────────────────────────────────────────────────────
    // Enter when inventory is small and LOB entropy is high (liquid conditions).
    // Direction follows filtered drift — a momentum signal on filtered price.
    if (std::abs(q_norm) < 0.3 && S_norm > 0.6) {
        double drift_thresh = sigma_norm_ * SIGMA_REF_BPS * 0.5;  // half-sigma in bps
        bool long_signal  = d_filt >  drift_thresh / 1e4;
        bool short_signal = d_filt < -drift_thresh / 1e4;

        if (long_signal || short_signal) {
            Side entry_side = long_signal ? Side::BUY : Side::SELL;
            double edge     = spread_n * 0.25;
            double px       = (entry_side == Side::BUY)
                              ? (p_filt + edge)
                              : (p_filt - edge);

            StrategyOrder o;
            o.symbol          = tick.symbol;
            o.side            = entry_side;
            o.type            = OrderType::LIMIT;
            o.price           = to_fixed_price(px * PRICE_NORM);
            o.quantity        = static_cast<Quantity>(ENTRY_LOT);
            o.client_order_id = tick.sequence * 4;

            pending_[o.client_order_id] = entry_side;
            // Optimistically track inventory at submission time.
            // onOrderResponse updates this again if fills are actually reported.
            inventory_ += (entry_side == Side::BUY) ? ENTRY_LOT : -ENTRY_LOT;
            submit_order(o);
            ++orders_sent_;
        }
    }
    record_timestamp("entry_done");

    // ── Step 5: HJB exit surface ──────────────────────────────────────────
    // Exit when V(q_norm, λ_norm) + F > threshold, i.e. when the value of
    // holding the position exceeds the estimated liquidation cost.
    if (std::abs(q_norm) > 0.05 &&
        hjb_.triggers_exit(q_norm, lam_norm, last_free_e_)) {

        // Lot size from HJB gradient: |∂V/∂q| is proportional to exit urgency.
        double grad    = std::abs(hjb_.grad_q(q_norm, lam_norm));
        static constexpr double HJB_A = 0.30;  // matches HJBExitSurface::Params::A
        double urgency = std::min(1.0, grad / (2.0 * HJB_A + 1e-8));
        double lot     = std::max(1.0, EXIT_LOT * urgency);
        lot            = std::min(lot, std::abs(inventory_));

        // Post passively at filtered price ± small edge to avoid adverse crossing.
        double edge    = spread_n * 0.1;
        double exit_px = (exit_side == Side::SELL)
                         ? (p_filt - edge)
                         : (p_filt + edge);

        StrategyOrder o;
        o.symbol          = tick.symbol;
        o.side            = exit_side;
        o.type            = OrderType::LIMIT;
        o.price           = to_fixed_price(exit_px * PRICE_NORM);
        o.quantity        = static_cast<Quantity>(lot);
        o.client_order_id = tick.sequence * 4 + 1;

        pending_[o.client_order_id] = exit_side;
        submit_order(o);
        ++orders_sent_;
        ++exits_fired_;
    }

    record_timestamp("hjb_exit_done");
}

void LOBExitStrategy::onOrderResponse(const OrderResponse& resp) {
    // Entry orders were already credited optimistically at submission.
    // Only process exit-order fills (odd client_order_ids = tick.sequence*4+1).
    if (resp.status != OrderStatus::FILLED &&
        resp.status != OrderStatus::PARTIALLY_FILLED) return;

    auto it = pending_.find(resp.client_order_id);
    if (it == pending_.end()) return;

    // Exit fills reduce inventory; entry fills were already credited.
    bool is_exit = (resp.client_order_id % 4 == 1);
    if (is_exit) {
        double qty = static_cast<double>(resp.fill_quantity);
        inventory_ += (it->second == Side::BUY) ? qty : -qty;
    }

    if (resp.status == OrderStatus::FILLED)
        pending_.erase(it);
}

void LOBExitStrategy::onShutdown() {
    std::cout << "[LOBExitStrategy] Shutdown\n"
              << "  Ticks processed:  " << tick_count_   << "\n"
              << "  Orders sent:      " << orders_sent_  << "\n"
              << "  Exits triggered:  " << exits_fired_  << "\n"
              << "  Final inventory:  " << inventory_    << "\n"
              << "  Last free energy: " << last_free_e_  << "\n"
              << "  Filtered vol (σ): " << sigma_norm_ * SIGMA_REF_BPS << " bps\n"
              << "  Hawkes λ at stop: " << hawkes_.intensity(last_t_sec_) << "/sec\n";
}

std::unique_ptr<UserStrategy> create_lob_exit_strategy() {
    return std::make_unique<LOBExitStrategy>();
}

} // namespace hft
