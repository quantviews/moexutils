import marimo

__generated_with = "0.18.4"
app = marimo.App(
    width="full",
    layout_file="layouts/blackjack_marimo.slides.json",
)


@app.cell
def _():
    import marimo as mo
    import random
    import math
    from dataclasses import dataclass
    from typing import List, Tuple
    import pandas as pd
    import matplotlib.pyplot as plt
    return List, Tuple, dataclass, math, mo, pd, plt, random


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Blackjack: симуляция в стиле Эда Торпа (Hi-Lo система)

    ## Теоретическая основа

    Этот ноутбук демонстрирует **card counting** (подсчета карт) — стратегию, разработанную Эдом Торпом в 1960-х годах.

    ### Ключевые концепции:

    1. **Hi-Lo система подсчёта:**
       - Низкие карты (2-6): +1. Эти карты наиболее полезны для дилера, так как он обязян добирать (12-16). Низкие карты позволяют ему набрать сумму и не перебрать. Когда эти карты выходят из игры, то вероятность того, что дилер переберет в будущем расчет -> матожидание игрока увеличивается
       - Средние (7-9): 0.
       - Высокие (10, J, Q, K, A): -1. Эти карты выгодны игроку. Когда они выходят из игры, то колода становится беднее. Шансы игрока снижаются.
       - Положительный счёт означает преимущество игрока

    2. **True Count (TC):**
       - TC = Running Count / Оставшиеся колоды
       - Нормализует счёт относительно глубины колоды
       - Показывает реальное преимущество

    3. **Базовая стратегия:**
       - Оптимальные решения без подсчёта карта, просто действуем по таблице
       - Минимизирует  "преимущества дома" (house edge) до ~0.5%

    4. **Spread ставок:**
       - Увеличение ставки при положительном TC
       - Управление рисками через размер ставки

    ### Связь с финансами:
    - **Edge**: преимущество игрока (аналог alpha в финансах)
    - **Bankroll management**: управление капиталом
    - **Risk/reward**: баланс риска и доходности
    - **Monte Carlo**: симуляция для оценки стратегии
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Параметры симуляции

    **Число раздач**: количество симулируемых раундов (больше = точнее, но дольше)

    **Кол-во колод**: стандартно 6-8 колод в казино (больше колод = сложнее считать)

    **Penetration**: доля колоды, которая раздаётся перед перетасовкой (0.75 = 75%)
    - Выше penetration → больше возможностей использовать счёт
    - Типично 0.75-0.85 в реальных казино

    **Базовая ставка**: минимальная ставка (единица измерения)

    **Seed**: для воспроизводимости результатов
    """)
    return


@app.cell
def _(mo):
    n_rounds = mo.ui.number(value=50000, step=10000, label="Число раздач", start=1000)
    n_decks = mo.ui.slider(1, 8, value=6, label="Кол-во колод")
    penetration = mo.ui.slider(0.5, 0.95, value=0.75, step=0.01, label="Penetration")
    base_bet = mo.ui.number(value=10.0, step=1.0, label="Базовая ставка", start=1.0)
    seed = mo.ui.number(value=123, step=1, label="Seed", start=0)
    return base_bet, n_decks, n_rounds, penetration, seed


@app.cell(hide_code=True)
def _(base_bet, mo, n_decks, n_rounds, penetration, seed):
    mo.md(f"""
    ### Настройки симуляции

    {mo.hstack([n_rounds, n_decks, penetration, base_bet, seed], justify="start", gap=1)}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Spread ставок (управление размером ставки)

    **Стратегия**: увеличиваем ставку при положительном True Count

    **Пример**:
    - TC < 1.0: ставка = базовая × 1
    - TC ≥ 1.0: ставка = базовая × 2
    - TC ≥ 2.0: ставка = базовая × 4
    - TC ≥ 3.0: ставка = базовая × 8

    **Важно**: больший spread даёт больше прибыли, но увеличивает волатильность и риск обнаружения в казино.
    """)
    return


@app.cell
def _(mo):
    tc1 = mo.ui.number(value=1.0, step=0.5, label="Порог TC 1 (>=)", start=0.0)
    m1 = mo.ui.number(value=2.0, step=0.5, label="Множитель 1", start=1.0)
    tc2 = mo.ui.number(value=2.0, step=0.5, label="Порог TC 2 (>=)", start=0.0)
    m2 = mo.ui.number(value=4.0, step=0.5, label="Множитель 2", start=1.0)
    tc3 = mo.ui.number(value=3.0, step=0.5, label="Порог TC 3 (>=)", start=0.0)
    m3 = mo.ui.number(value=8.0, step=0.5, label="Множитель 3", start=1.0)
    return m1, m2, m3, tc1, tc2, tc3


@app.cell(hide_code=True)
def _(m1, m2, m3, mo, tc1, tc2, tc3):
    mo.md(f"""
    1542016### Настройки spread

    {mo.hstack([tc1, m1, tc2, m2, tc3, m3], justify="start", gap=1)}
    """)
    return


@app.cell
def _(mo):
    run_btn = mo.ui.run_button(label="Запустить симуляцию")
    mo.md(f"## Запуск\n\n{run_btn}")
    return (run_btn,)


@app.cell(hide_code=True)
def _():
    RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

    def card_value(rank: str) -> int:
        if rank == "A":
            return 11
        if rank in ("J", "Q", "K"):
            return 10
        return int(rank)

    def hi_lo_tag(rank: str) -> int:
        if rank in ("2", "3", "4", "5", "6"):
            return +1
        if rank in ("7", "8", "9"):
            return 0
        return -1
    return RANKS, card_value, hi_lo_tag


@app.cell(hide_code=True)
def _(List, Tuple, card_value):
    def hand_totals(cards: List[str]) -> Tuple[int, bool]:
        total = 0
        aces = 0
        for r in cards:
            total += card_value(r)
            if r == "A":
                aces += 1

        while total > 21 and aces > 0:
            total -= 10
            aces -= 1

        soft = False
        if "A" in cards:
            hard_total = sum(1 if r == "A" else card_value(r) for r in cards)
            if hard_total != total and total <= 21:
                soft = True

        return total, soft

    def is_blackjack(cards: List[str]) -> bool:
        return len(cards) == 2 and ("A" in cards) and any(r in ("10", "J", "Q", "K") for r in cards)

    def dealer_up_value(rank: str) -> int:
        if rank == "A":
            return 11
        if rank in ("10", "J", "Q", "K"):
            return 10
        return int(rank)

    def is_pair(cards: List[str]) -> bool:
        if len(cards) != 2:
            return False
        a, b = cards
        if a in ("10", "J", "Q", "K") and b in ("10", "J", "Q", "K"):
            return True
        return a == b

    def pair_rank(cards: List[str]) -> str:
        a, b = cards
        if a in ("10", "J", "Q", "K") and b in ("10", "J", "Q", "K"):
            return "10"
        return a
    return dealer_up_value, hand_totals, is_blackjack, is_pair, pair_rank


@app.cell(hide_code=True)
def _(List, RANKS, dataclass, hi_lo_tag, random):
    @dataclass
    class Shoe:
        n_decks: int = 6
        penetration: float = 0.75
        rng: random.Random = None

        def __post_init__(self):
            if self.rng is None:
                self.rng = random.Random(42)
            self.cards: List[str] = []
            self.running_count: int = 0
            self._initial_size = 52 * self.n_decks
            self.shuffle()

        def shuffle(self):
            self.cards = []
            for _ in range(self.n_decks):
                for r in RANKS:
                    self.cards.extend([r] * 4)
            self.rng.shuffle(self.cards)
            self.running_count = 0

        def decks_remaining(self) -> float:
            return max(len(self.cards) / 52.0, 0.01)

        def true_count(self) -> float:
            return self.running_count / self.decks_remaining()

        def maybe_reshuffle(self):
            dealt = self._initial_size - len(self.cards)
            if dealt / self._initial_size >= self.penetration:
                self.shuffle()

        def draw(self) -> str:
            if not self.cards:
                self.shuffle()
            c = self.cards.pop()
            self.running_count += hi_lo_tag(c)
            return c
    return (Shoe,)


@app.cell(hide_code=True)
def _(List, dealer_up_value, hand_totals, is_pair, pair_rank):
    def basic_strategy_action(cards: List[str], dealer_up: str, can_split: bool, can_double: bool) -> str:
        up = dealer_up_value(dealer_up)

        # Splits (S17 + DAS, приближение)
        if can_split and is_pair(cards):
            pr = pair_rank(cards)
            if pr == "A":
                return "P"
            if pr == "8":
                return "P"
            if pr in ("2", "3"):
                return "P" if up in (2, 3, 4, 5, 6, 7) else "H"
            if pr == "4":
                return "P" if up in (5, 6) else "H"
            if pr == "6":
                return "P" if up in (2, 3, 4, 5, 6) else "H"
            if pr == "7":
                return "P" if up in (2, 3, 4, 5, 6, 7) else "H"
            if pr == "9":
                return "P" if up in (2, 3, 4, 5, 6, 8, 9) else "S"
            if pr == "10":
                return "S"

        total, soft = hand_totals(cards)

        # Soft totals
        if soft:
            if total <= 17:
                if can_double:
                    if total in (13, 14) and up in (5, 6):
                        return "D"
                    if total in (15, 16) and up in (4, 5, 6):
                        return "D"
                    if total == 17 and up in (3, 4, 5, 6):
                        return "D"
                return "H"
            if total == 18:
                if can_double and up in (3, 4, 5, 6):
                    return "D"
                if up in (9, 10, 11):
                    return "H"
                return "S"
            return "S"

        # Hard totals
        if total <= 8:
            return "H"
        if total == 9:
            if can_double and up in (3, 4, 5, 6):
                return "D"
            return "H"
        if total == 10:
            if can_double and up in (2, 3, 4, 5, 6, 7, 8, 9):
                return "D"
            return "H"
        if total == 11:
            if can_double:
                return "D"
            return "H"
        if total == 12:
            return "S" if up in (4, 5, 6) else "H"
        if total in (13, 14, 15, 16):
            return "S" if up in (2, 3, 4, 5, 6) else "H"
        return "S"
    return (basic_strategy_action,)


@app.cell(hide_code=True)
def _(List, Shoe, dataclass, hand_totals):
    @dataclass
    class HandState:
        cards: List[str]
        bet: float
        is_split_aces: bool = False
        doubled: bool = False

    def dealer_play(shoe: Shoe, dealer_cards: List[str], stand_on_soft_17: bool = True) -> List[str]:
        while True:
            total, soft = hand_totals(dealer_cards)
            if total > 21:
                return dealer_cards
            if total < 17:
                dealer_cards.append(shoe.draw())
                continue
            if total == 17 and soft and not stand_on_soft_17:
                dealer_cards.append(shoe.draw())
                continue
            return dealer_cards

    def resolve_hand(player_total: int, dealer_total: int) -> int:
        if player_total > 21:
            return -1
        if dealer_total > 21:
            return +1
        if player_total > dealer_total:
            return +1
        if player_total < dealer_total:
            return -1
        return 0
    return HandState, dealer_play, resolve_hand


@app.cell(hide_code=True)
def _(
    HandState,
    List,
    Shoe,
    Tuple,
    basic_strategy_action,
    dealer_play,
    hand_totals,
    is_blackjack,
    resolve_hand,
):
    def pick_bet_multiplier(true_count: float, spread: List[Tuple[float, float]]) -> float:
        mult = 1.0
        for thr, m in spread:
            if true_count >= thr:
                mult = m
        return mult

    def play_round(shoe: Shoe, base_bet_val: float, spread: List[Tuple[float, float]]):
        shoe.maybe_reshuffle()

        tc_before = shoe.true_count()
        bet0 = base_bet_val * pick_bet_multiplier(tc_before, spread)

        player = [shoe.draw(), shoe.draw()]
        dealer = [shoe.draw(), shoe.draw()]
        dealer_up = dealer[0]

        amount_wagered = bet0

        if dealer_up in ("A", "10", "J", "Q", "K") and is_blackjack(dealer):
            if is_blackjack(player):
                return 0.0, amount_wagered, tc_before, bet0
            return -bet0, amount_wagered, tc_before, bet0

        if is_blackjack(player):
            return 1.5 * bet0, amount_wagered, tc_before, bet0

        hands: List[HandState] = [HandState(cards=player, bet=bet0)]
        split_count = 0
        i = 0

        while i < len(hands):
            h = hands[i]

            if h.is_split_aces:
                i += 1
                continue

            while True:
                total, _ = hand_totals(h.cards)
                if total > 21:
                    break

                can_split = (len(h.cards) == 2 and split_count < 3)
                can_double = (len(h.cards) == 2 and not h.doubled)

                act = basic_strategy_action(h.cards, dealer_up, can_split=can_split, can_double=can_double)

                if act == "P" and can_split and len(h.cards) == 2:
                    split_count += 1
                    c1, c2 = h.cards
                    h.cards = [c1, shoe.draw()]
                    h.is_split_aces = (c1 == "A")

                    new_hand = HandState(cards=[c2, shoe.draw()], bet=h.bet, is_split_aces=(c2 == "A"))
                    hands.append(new_hand)

                    amount_wagered += h.bet
                    break

                if act == "D" and can_double:
                    h.bet *= 2
                    h.doubled = True
                    amount_wagered += h.bet / 2
                    h.cards.append(shoe.draw())
                    break

                if act == "S":
                    break

                h.cards.append(shoe.draw())

            i += 1

        dealer = dealer_play(shoe, dealer, stand_on_soft_17=True)
        dealer_total, _ = hand_totals(dealer)

        profit = 0.0
        for h in hands:
            pt, _ = hand_totals(h.cards)
            profit += resolve_hand(pt, dealer_total) * h.bet

        return profit, amount_wagered, tc_before, bet0
    return (play_round,)


@app.cell(hide_code=True)
def _(
    Shoe,
    base_bet,
    m1,
    m2,
    m3,
    math,
    mo,
    n_decks,
    n_rounds,
    pd,
    penetration,
    play_round,
    random,
    run_btn,
    seed,
    tc1,
    tc2,
    tc3,
):
    _df = None
    _summary = None

    if run_btn.value:
        _rng = random.Random(int(seed.value))
        _shoe = Shoe(n_decks=int(n_decks.value), penetration=float(penetration.value), rng=_rng)

        _spread = sorted(
            [
                (float(tc1.value), float(m1.value)),
                (float(tc2.value), float(m2.value)),
                (float(tc3.value), float(m3.value)),
            ],
            key=lambda x: x[0],
        )

        _rows = []
        _total_profit = 0.0
        _total_wagered = 0.0
        _N = int(n_rounds.value)

        for _t in range(_N):
            _p, _w, _tc_b, _bet0 = play_round(_shoe, float(base_bet.value), _spread)
            _total_profit += _p
            _total_wagered += _w
            _rows.append((_t + 1, _p, _w, _tc_b, _bet0, _total_profit))

        _df = pd.DataFrame(_rows, columns=["hand", "profit", "wagered", "true_count", "bet", "cum_profit"])

        _mu = float(_df["profit"].mean())
        _sd = float(_df["profit"].std(ddof=1))
        _se = _sd / math.sqrt(_N)
        _ci_low = _mu - 1.96 * _se
        _ci_high = _mu + 1.96 * _se
        _roi = _total_profit / _total_wagered if _total_wagered > 0 else float("nan")

        # Дополнительные метрики
        _wins = (_df["profit"] > 0).sum()
        _losses = (_df["profit"] < 0).sum()
        _pushes = (_df["profit"] == 0).sum()
        _win_rate = _wins / _N if _N > 0 else 0.0

        _avg_win = float(_df[_df["profit"] > 0]["profit"].mean()) if _wins > 0 else 0.0
        _avg_loss = float(_df[_df["profit"] < 0]["profit"].mean()) if _losses > 0 else 0.0

        # Drawdown
        _running_max = _df["cum_profit"].expanding().max()
        _drawdown = _df["cum_profit"] - _running_max
        _max_drawdown = float(_drawdown.min())

        # Sharpe ratio (annualized, assuming ~100 hands per hour)
        _sharpe = (_mu / _sd * math.sqrt(100 * 2000)) if _sd > 0 else 0.0

        # Profit by true count ranges
        _df["tc_range"] = pd.cut(_df["true_count"], bins=[-10, -1, 0, 1, 2, 3, 10],
                                  labels=["TC<-1", "-1≤TC<0", "0≤TC<1", "1≤TC<2", "2≤TC<3", "TC≥3"])
        _profit_by_tc = _df.groupby("tc_range", observed=False)["profit"].agg(["mean", "count"]).to_dict("index")

        _summary = {
            "rounds": _N,
            "total_profit": _total_profit,
            "total_wagered": _total_wagered,
            "roi_percent": 100.0 * _roi,
            "mean_profit_per_hand": _mu,
            "sd_profit_per_hand": _sd,
            "ci95_low": _ci_low,
            "ci95_high": _ci_high,
            "spread": _spread,
            "win_rate": _win_rate,
            "wins": _wins,
            "losses": _losses,
            "pushes": _pushes,
            "avg_win": _avg_win,
            "avg_loss": _avg_loss,
            "max_drawdown": _max_drawdown,
            "sharpe_ratio": _sharpe,
            "profit_by_tc": _profit_by_tc,
        }

    if _summary is None:
        mo.output.replace(mo.callout("Нажмите кнопку «Запустить симуляцию» выше.", kind="info"))
    else:
        # Определяем цвет для прибыли
        _profit_color = "green" if _summary["total_profit"] > 0 else "red"
        _roi_color = "green" if _summary["roi_percent"] > 0 else "red"
        _edge_status = "success" if _summary["ci95_low"] > 0 else ("danger" if _summary["ci95_high"] < 0 else "warn")

        # Главный индикатор результата
        if _summary["total_profit"] > 0:
            _result_callout = mo.callout(
                mo.md(f"### 🎰 Прибыль: **+{_summary['total_profit']:.2f}** (ROI: {_summary['roi_percent']:.3f}%)"),
                kind="success"
            )
        else:
            _result_callout = mo.callout(
                mo.md(f"### 🎰 Убыток: **{_summary['total_profit']:.2f}** (ROI: {_summary['roi_percent']:.3f}%)"),
                kind="danger"
            )

        # Статистическая значимость
        if _summary["ci95_low"] > 0:
            _stat_callout = mo.callout(
                mo.md("✅ **Статистически значимое преимущество** — 95% ДИ полностью выше нуля"),
                kind="success"
            )
        elif _summary["ci95_high"] < 0:
            _stat_callout = mo.callout(
                mo.md("❌ **Статистически значимый проигрыш** — 95% ДИ полностью ниже нуля"),
                kind="danger"
            )
        else:
            _stat_callout = mo.callout(
                mo.md("⚠️ **Результат не значим** — 95% ДИ содержит ноль"),
                kind="warn"
            )

        # Основные статистики в виде карточек
        _stats_row1 = mo.hstack([
            mo.stat(
                value=f"{_summary['rounds']:,}",
                label="Раздач",
                bordered=True
            ),
            mo.stat(
                value=f"{_summary['total_wagered']:,.0f}",
                label="Сумма ставок",
                bordered=True
            ),
            mo.stat(
                value=f"{_summary['roi_percent']:.3f}%",
                label="ROI",
                bordered=True
            ),
            mo.stat(
                value=f"{_summary['sharpe_ratio']:.2f}",
                label="Sharpe Ratio",
                bordered=True
            ),
        ], justify="space-around", gap=1)

        _stats_row2 = mo.hstack([
            mo.stat(
                value=f"{_summary['win_rate']:.1%}",
                label="Win Rate",
                bordered=True
            ),
            mo.stat(
                value=f"{_summary['wins']:,}",
                label="Выигрышей",
                bordered=True
            ),
            mo.stat(
                value=f"{_summary['losses']:,}",
                label="Проигрышей",
                bordered=True
            ),
            mo.stat(
                value=f"{_summary['pushes']:,}",
                label="Ничьих",
                bordered=True
            ),
        ], justify="space-around", gap=1)

        _stats_row3 = mo.hstack([
            mo.stat(
                value=f"{_summary['mean_profit_per_hand']:.4f}",
                label="Средняя прибыль/руку",
                bordered=True
            ),
            mo.stat(
                value=f"{_summary['sd_profit_per_hand']:.2f}",
                label="Станд. откл.",
                bordered=True
            ),
            mo.stat(
                value=f"{_summary['max_drawdown']:.2f}",
                label="Max Drawdown",
                bordered=True
            ),
            mo.stat(
                value=f"{abs(_summary['avg_win'] / _summary['avg_loss']):.2f}",
                label="Win/Loss Ratio",
                bordered=True
            ),
        ], justify="space-around", gap=1)

        # Детали в аккордеоне
        _details = mo.accordion({
            "📊 Детальная статистика": mo.md(f"""
    | Метрика | Значение |
    |---------|----------|
    | Средняя прибыль на руку | {_summary['mean_profit_per_hand']:.6f} |
    | Стандартное отклонение | {_summary['sd_profit_per_hand']:.4f} |
    | 95% ДИ (нижняя граница) | {_summary['ci95_low']:.6f} |
    | 95% ДИ (верхняя граница) | {_summary['ci95_high']:.6f} |
    | Средний выигрыш | {_summary['avg_win']:.2f} |
    | Средний проигрыш | {_summary['avg_loss']:.2f} |
            """),
            "⚙️ Настройки spread": mo.md(f"""
    | Порог TC | Множитель ставки |
    |----------|------------------|
    | TC < {_summary['spread'][0][0]:.1f} | ×1 (базовая) |
    | TC ≥ {_summary['spread'][0][0]:.1f} | ×{_summary['spread'][0][1]:.1f} |
    | TC ≥ {_summary['spread'][1][0]:.1f} | ×{_summary['spread'][1][1]:.1f} |
    | TC ≥ {_summary['spread'][2][0]:.1f} | ×{_summary['spread'][2][1]:.1f} |
            """),
            "📈 Интерпретация результатов": mo.md("""
    - **ROI > 0%**: стратегия прибыльна в долгосрочной перспективе
    - **95% ДИ не содержит 0**: статистически значимое преимущество
    - **Sharpe > 1**: хорошая риск-скорректированная доходность
    - **Sharpe > 2**: отличная доходность
    - **Max Drawdown**: максимальная просадка от пика — показывает риск
            """),
        })

        # Собираем всё вместе
        _output = mo.vstack([
            mo.md("## 🎲 Результаты симуляции"),
            _result_callout,
            _stat_callout,
            mo.md("### Ключевые метрики"),
            _stats_row1,
            mo.md("### Win/Loss статистика"),
            _stats_row2,
            mo.md("### Риск и доходность"),
            _stats_row3,
            _details,
        ], gap=1)

        mo.output.replace(_output)

    _df, _summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Визуализация результатов
    """)
    return


@app.cell(hide_code=True)
def _(
    Shoe,
    base_bet,
    m1,
    m2,
    m3,
    mo,
    n_decks,
    n_rounds,
    pd,
    penetration,
    play_round,
    plt,
    random,
    run_btn,
    seed,
    tc1,
    tc2,
    tc3,
):
    # Re-run simulation for charts
    _chart1 = None
    if run_btn.value:
        _rng2 = random.Random(int(seed.value))
        _shoe2 = Shoe(n_decks=int(n_decks.value), penetration=float(penetration.value), rng=_rng2)
        _spread2 = sorted([
            (float(tc1.value), float(m1.value)),
            (float(tc2.value), float(m2.value)),
            (float(tc3.value), float(m3.value)),
        ], key=lambda x: x[0])

        _rows2 = []
        _total_profit2 = 0.0
        _N2 = int(n_rounds.value)
        for _t in range(_N2):
            _p, _w, _tc_b, _bet0 = play_round(_shoe2, float(base_bet.value), _spread2)
            _total_profit2 += _p
            _rows2.append((_t + 1, _p, _w, _tc_b, _bet0, _total_profit2))

        _df2 = pd.DataFrame(_rows2, columns=["hand", "profit", "wagered", "true_count", "bet", "cum_profit"])

        # График 1: Накопленная прибыль
        _fig1, _ax1 = plt.subplots(figsize=(12, 5))
        _ax1.plot(_df2["hand"], _df2["cum_profit"], linewidth=0.8, alpha=0.8)
        _ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
        _ax1.set_title("Накопленная прибыль (Equity Curve)", fontsize=14, fontweight='bold')
        _ax1.set_xlabel("Номер раздачи")
        _ax1.set_ylabel("Накопленная прибыль")
        _ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        _chart1 = _fig1
    _chart1 if _chart1 else mo.md("")
    return


@app.cell(hide_code=True)
def _(
    Shoe,
    base_bet,
    m1,
    m2,
    m3,
    mo,
    n_decks,
    n_rounds,
    pd,
    penetration,
    play_round,
    plt,
    random,
    run_btn,
    seed,
    tc1,
    tc2,
    tc3,
):
    # Drawdown chart
    _chart2 = None
    if run_btn.value:
        _rng3 = random.Random(int(seed.value))
        _shoe3 = Shoe(n_decks=int(n_decks.value), penetration=float(penetration.value), rng=_rng3)
        _spread3 = sorted([
            (float(tc1.value), float(m1.value)),
            (float(tc2.value), float(m2.value)),
            (float(tc3.value), float(m3.value)),
        ], key=lambda x: x[0])

        _rows3 = []
        _total3 = 0.0
        _N3 = int(n_rounds.value)
        for _t in range(_N3):
            _p, _w, _tc_b, _bet0 = play_round(_shoe3, float(base_bet.value), _spread3)
            _total3 += _p
            _rows3.append((_t + 1, _p, _w, _tc_b, _bet0, _total3))

        _df3 = pd.DataFrame(_rows3, columns=["hand", "profit", "wagered", "true_count", "bet", "cum_profit"])

        _running_max3 = _df3["cum_profit"].expanding().max()
        _drawdown3 = _df3["cum_profit"] - _running_max3

        _fig_dd, _ax_dd = plt.subplots(figsize=(12, 4))
        _ax_dd.fill_between(_df3["hand"], _drawdown3, 0, alpha=0.5, color='red')
        _ax_dd.set_title("Drawdown (просадка от максимума)", fontsize=14, fontweight='bold')
        _ax_dd.set_xlabel("Номер раздачи")
        _ax_dd.set_ylabel("Drawdown")
        _ax_dd.grid(True, alpha=0.3)
        plt.tight_layout()
        _chart2 = _fig_dd
    _chart2 if _chart2 else mo.md("")
    return


@app.cell(hide_code=True)
def _(
    Shoe,
    base_bet,
    m1,
    m2,
    m3,
    mo,
    n_decks,
    n_rounds,
    pd,
    penetration,
    play_round,
    plt,
    random,
    run_btn,
    seed,
    tc1,
    tc2,
    tc3,
):
    # Distribution chart
    _chart3 = None
    if run_btn.value:
        _rng4 = random.Random(int(seed.value))
        _shoe4 = Shoe(n_decks=int(n_decks.value), penetration=float(penetration.value), rng=_rng4)
        _spread4 = sorted([
            (float(tc1.value), float(m1.value)),
            (float(tc2.value), float(m2.value)),
            (float(tc3.value), float(m3.value)),
        ], key=lambda x: x[0])

        _rows4 = []
        _total4 = 0.0
        _N4 = int(n_rounds.value)
        for _t in range(_N4):
            _p, _w, _tc_b, _bet0 = play_round(_shoe4, float(base_bet.value), _spread4)
            _total4 += _p
            _rows4.append((_t + 1, _p, _w, _tc_b, _bet0, _total4))

        _df4 = pd.DataFrame(_rows4, columns=["hand", "profit", "wagered", "true_count", "bet", "cum_profit"])
        _mean4 = _df4["profit"].mean()

        _fig2, _ax2 = plt.subplots(figsize=(10, 6))
        _ax2.hist(_df4["profit"], bins=51, edgecolor='black', alpha=0.7)
        _ax2.axvline(x=0, color='r', linestyle='--', linewidth=2, alpha=0.7, label='Безубыточность')
        _ax2.axvline(x=_mean4, color='g', linestyle='--', linewidth=2, alpha=0.7, label=f'Среднее: {_mean4:.4f}')
        _ax2.set_title("Распределение прибыли на руку", fontsize=14, fontweight='bold')
        _ax2.set_xlabel("Прибыль на руку")
        _ax2.set_ylabel("Частота")
        _ax2.legend()
        _ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        _chart3 = _fig2
    _chart3 if _chart3 else mo.md("")
    return


@app.cell(hide_code=True)
def _(
    Shoe,
    base_bet,
    m1,
    m2,
    m3,
    mo,
    n_decks,
    n_rounds,
    pd,
    penetration,
    play_round,
    plt,
    random,
    run_btn,
    seed,
    tc1,
    tc2,
    tc3,
):
    # True count distribution
    _chart4 = None
    if run_btn.value:
        _rng5 = random.Random(int(seed.value))
        _shoe5 = Shoe(n_decks=int(n_decks.value), penetration=float(penetration.value), rng=_rng5)
        _spread5 = sorted([
            (float(tc1.value), float(m1.value)),
            (float(tc2.value), float(m2.value)),
            (float(tc3.value), float(m3.value)),
        ], key=lambda x: x[0])

        _rows5 = []
        _total5 = 0.0
        _N5 = int(n_rounds.value)
        for _t in range(_N5):
            _p, _w, _tc_b, _bet0 = play_round(_shoe5, float(base_bet.value), _spread5)
            _total5 += _p
            _rows5.append((_t + 1, _p, _w, _tc_b, _bet0, _total5))

        _df5 = pd.DataFrame(_rows5, columns=["hand", "profit", "wagered", "true_count", "bet", "cum_profit"])

        _fig4, _ax4 = plt.subplots(figsize=(10, 6))
        _ax4.hist(_df5["true_count"], bins=50, edgecolor='black', alpha=0.7)
        _ax4.axvline(x=0, color='r', linestyle='--', linewidth=2, alpha=0.7, label='TC = 0')
        _ax4.set_title("Распределение True Count", fontsize=14, fontweight='bold')
        _ax4.set_xlabel("True Count")
        _ax4.set_ylabel("Частота")
        _ax4.legend()
        _ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        _chart4 = _fig4
    _chart4 if _chart4 else mo.md("")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Рекомендации для экспериментов

    ### Что попробовать:

    1. **Влияние penetration**:
       - Уменьшите penetration до 0.5 → меньше возможностей использовать счёт
       - Увеличьте до 0.9 → больше возможностей, но нереалистично для казино

    2. **Влияние spread**:
       - Уменьшите множители → меньше прибыль, но меньше риск
       - Увеличьте spread → больше прибыль, но выше волатильность

    3. **Влияние числа колод**:
       - 1 колода → легче считать, но редко в казино
       - 8 колод → сложнее, но реалистичнее

    4. **Статистическая значимость**:
       - Увеличьте число раздач до 100,000+ для более точных результатов
       - Обратите внимание на 95% ДИ — он должен не содержать 0 для прибыльной стратегии

    ### Вопросы для размышления:

    - Почему стратегия работает только при положительном True Count?
    - Как связаны spread и волатильность результатов?
    - Что показывает Sharpe ratio и почему он важен?
    - Как бы вы применили эти концепции к торговле на финансовых рынках?

    ### Связь с финансами:

    - **Edge** (преимущество) = положительное математическое ожидание
    - **Bankroll management** = управление размером позиции
    - **Drawdown** = просадка капитала (критично для риск-менеджмента)
    - **Sharpe ratio** = метрика риск-скорректированной доходности
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Сравнение с финансами

    ## Поиска альфа

    В Блэкджеке:

    - Базовая стратегия дает вам результат, близкий к рыночному (Beta) — вы медленно теряете на комиссии казино (House Edge ~0.5%).
    - Подсчет карт (Card Counting) — это поиск ситуаций, когда вероятностное пространство смещается в вашу пользу. Высокий True Count означает, что рынок "неэффективен" и недооценивает вероятность вашего выигрыша. Это и есть  Alpha (преимущество над рынком).

    На рынке акций:

    - Индексное инвестирование  — это игра по "базовой стратегии". Вы получаете среднюю доходность рынка.
    - Активные операции  — это поиск временных неэффективностей (mispricing). Например, стратегия статистического арбитража ищет пары акций, чья корреляция временно нарушилась, аналогично тому, как счетчик ищет колоду, насыщенную десятками.

    ## Управление размером позиции (position sizing)

    -  Если ваш edge (матожидание преимущества) высок (высокий True Count или сильный сигнал модели) — вы увеличиваете ставку (размер лота/плечо).
    -  Если edge отрицательный или нулевой — вы держите минимум (или выходите в деньги)
    - Использование чрезмерного кредитного плеча к margin call, даже если ваша стратегия долгосрочно прибыльна. История фонда LTCM — классический пример игнорирования "риска разорения" при правильной математической модели.

    ## Гипотеза эффективного рынка (EMH)

    - В Блэкджеке если казино использует машинку для непрерывного перемешивания, каждая раздача становится независимой. Прошлые данные не предсказывают будущие. Рынок становится абсолютно эффективным. В таких условиях подсчет карт невозможен.
    - На рынке акций кванты исходят из того, что рынок эффективен не всегда, и ищут "карманы неэффективности".

    ## Стационарность процесса

    - В Блэкджеке правила жесткие. В колоде всегда 52 карты. Вероятности стационарны и известны заранее.
    - На рынке распределение вероятностей **нестационарно**. Правила меняются, случаются "черные лебеди", корреляции распадаются во время кризисов. Модель, работавшая вчера (как подсчет карт), может перестать работать завтра из-за смены макроэкономической парадигмы. Рынок сложнее.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
