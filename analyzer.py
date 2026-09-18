# analyzer.py
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from scipy.stats import chisquare, entropy as scipy_entropy
import math


class MarketAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        self.all_numbers = [n for nums in self.df["numbers"] for n in nums]
        self.all_numbers = [int(x) for x in self.all_numbers if str(x).isdigit()]

    def frequency(self):
        return Counter(self.all_numbers)

    def hot_cold(self, top=5, window=15):
        recent = self.df.tail(window)
        rec_nums = [int(n) for nums in recent["numbers"] for n in nums if str(n).isdigit()]
        freq = Counter(rec_nums)
        hot = freq.most_common(top)
        cold = sorted(freq.items(), key=lambda x: x[1])[:top]
        return hot, cold

    def markov_chain(self):
        transitions = defaultdict(Counter)
        for nums in self.df["numbers"]:
            nums = [int(n) for n in nums if str(n).isdigit()]
            for i in range(len(nums) - 1):
                transitions[nums[i]][nums[i + 1]] += 1
        return transitions

    def predict_next(self, last_number):
        trans = self.markov_chain()
        if last_number in trans and trans[last_number]:
            total = sum(trans[last_number].values())
            probs = {n: round(c / total * 100, 1) for n, c in trans[last_number].most_common(3)}
            return probs
        return {}

    def entropy_score(self):
        freq = self.frequency()
        if not freq:
            return 0
        probs = np.array(list(freq.values())) / sum(freq.values())
        h = scipy_entropy(probs) / math.log2(10)
        return round(h, 3)

    def chi_square_test(self):
        freq = self.frequency()
        observed = [freq.get(str(i), 0) for i in range(10)]
        expected = [sum(observed) / 10] * 10
        if sum(observed) == 0:
            return 0, 1
        chi2, p = chisquare(observed, expected)
        return round(chi2, 2), round(p, 4)

    def streaks(self, min_len=2):
        streaks = []
        for nums in self.df["numbers"]:
            nums = [str(n) for n in nums]
            if not nums:
                continue
            current = [nums[0]]
            for n in nums[1:]:
                if n == current[-1]:
                    current.append(n)
                else:
                    if len(current) >= min_len:
                        streaks.append((current[0], len(current)))
                    current = [n]
            if len(current) >= min_len:
                streaks.append((current[0], len(current)))
        return streaks

    def weekday_pattern(self):
        if self.df.empty:
            return {}
        try:
            temp = self.df.copy()
            temp["dow"] = pd.to_datetime(temp["date"], errors="coerce").dt.day_name()
            result = {}
            for day, group in temp.dropna(subset=["dow"]).groupby("dow"):
                nums = [int(n) for lst in group["numbers"] for n in lst if str(n).isdigit()]
                if nums:
                    result[day] = Counter(nums).most_common(3)
            return result
        except Exception:
            return {}

    def gap_analysis(self):
        gaps = defaultdict(list)
        last_seen = {}
        for idx, nums in enumerate(self.df["numbers"]):
            for n in nums:
                n = str(n)
                if n in last_seen:
                    gaps[n].append(idx - last_seen[n])
                last_seen[n] = idx
        return {n: round(np.mean(g), 1) for n, g in gaps.items() if g}

    def even_odd_ratio(self):
        even = sum(1 for n in self.all_numbers if n % 2 == 0)
        odd = len(self.all_numbers) - even
        return even, odd

    def mood_score(self):
        if not self.all_numbers:
            return 0, "No data"
        ent = self.entropy_score()
        _, p_value = self.chi_square_test()
        freq = self.frequency()
        repeat_ratio = sum(c for c in freq.values() if c > 1) / len(self.all_numbers)
        pattern_score = (1 - ent) * 50 + (1 - min(p_value * 10, 1)) * 30 + repeat_ratio * 20
        pattern_score = round(min(pattern_score, 100), 2)
        if pattern_score > 65:
            mood = "Aggressive - Strong Pattern"
        elif pattern_score > 40:
            mood = "Neutral - Mixed Signals"
        else:
            mood = "Quiet - Highly Random"
        return pattern_score, mood

    def full_report(self):
        hot, cold = self.hot_cold()
        score, mood = self.mood_score()
        chi2, p = self.chi_square_test()
        even, odd = self.even_odd_ratio()
        last_num = self.all_numbers[-1] if self.all_numbers else None
        prediction = self.predict_next(str(last_num)) if last_num is not None else {}
        return {
            "total_entries": len(self.df),
            "total_numbers": len(self.all_numbers),
            "hot_numbers": hot,
            "cold_numbers": cold,
            "mood_score": score,
            "mood": mood,
            "entropy": self.entropy_score(),
            "chi_square": chi2,
            "p_value": p,
            "even_odd": (even, odd),
            "streaks": self.streaks()[-5:],
            "weekday_pattern": self.weekday_pattern(),
            "gaps": self.gap_analysis(),
            "last_number": last_num,
            "next_prediction": prediction,
            "frequency": dict(self.frequency()),
        }
