"""ทดสอบคลาสหลัก: python -m pytest tests"""
import pytest

from core.cards import Card, Deck
from core.encoders import PerPlayerEncoder, PresenceEncoder
from core.poker import DuplicateCardError, HandEvaluator, PokerTable
from core.predictors import RuleBasedPredictor


def hand(*codes):
    return [Card.from_code(c) for c in codes]


def test_deck_has_52_unique_cards():
    deck = Deck()
    assert len(deck) == 52 and len(set(deck)) == 52
    assert deck.index_of(Card.from_code("2S")) == 0
    assert deck.index_of(Card.from_code("AC")) == 51


def test_invalid_card_rejected():
    with pytest.raises(ValueError):
        Card.from_code("1X")


def test_pair_of_kings_beats_pair_of_twos_with_ace():
    ev = HandEvaluator()
    kings = ev.evaluate_5(hand("KS", "KD", "7C", "5H", "3D"))
    twos = ev.evaluate_5(hand("2S", "2D", "AC", "QH", "JD"))
    assert kings > twos


def test_wheel_is_lowest_straight():
    ev = HandEvaluator()
    wheel = ev.evaluate_5(hand("AS", "2D", "3C", "4H", "5D"))
    six_high = ev.evaluate_5(hand("2S", "3D", "4C", "5H", "6D"))
    assert ev.hand_name(wheel) == "Straight" and six_high > wheel


def test_duplicate_cards_raise():
    with pytest.raises(DuplicateCardError):
        PokerTable.from_codes([["AS", "KS"], ["AS", "2D"], ["3C", "4C"]], ["5H", "6H", "7H", "8H"])


def test_rule_based_split_pot():
    table = PokerTable.from_codes([["QC", "3D"], ["QD", "2C"], ["4S", "5S"]],
                                  ["AH", "AD", "KC", "9H", "7S"])
    result = RuleBasedPredictor().predict(table)
    assert result.probabilities == (0.5, 0.5, 0.0)


def test_encoders_feature_size():
    table = PokerTable.from_codes([["AS", "KS"], ["2D", "3D"], ["4C", "5C"]], ["6H", "7H", "8H", "9H"])
    assert PresenceEncoder().encode_table(table).sum() == 10
    assert PresenceEncoder().n_features == 52
    assert PerPlayerEncoder().encode_table(table).shape == (1, 208)
