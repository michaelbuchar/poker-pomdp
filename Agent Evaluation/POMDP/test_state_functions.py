import unittest
from treys import Card
from State_Generation_Methods import (
    lookup_hole_card_win_probability,
    calc_win_state_bucket,
    is_paired,
    is_board_connected,
    get_board_high_card_level,
    count_high_cards,
    get_board_state_features,
    get_rank_of_pair,
    get_number_of_paired_cards,
    get_flush_status,
    get_straight_status,
    get_full_house_status,
    get_player_board_features
)

class TestPokerFunctions(unittest.TestCase):

    def test_lookup_hole_card_win_probability(self):
        result1 = lookup_hole_card_win_probability(f"{Card.int_to_pretty_str(Card.new("As"))} {Card.int_to_pretty_str(Card.new("Kd"))}", "../Hole_Card_Probabilities/hole_card_simulation_results.csv")
        self.assertIsNotNone(result1)
        self.assertGreater(result1, 0)
        result2 = lookup_hole_card_win_probability(f"{Card.int_to_pretty_str(Card.new("Kd"))} {Card.int_to_pretty_str(Card.new("As"))}", "../Hole_Card_Probabilities/hole_card_simulation_results.csv")
        self.assertIsNotNone(result2)
        self.assertGreater(result2, 0)

    def test_calc_win_state_bucket(self):
        self.assertEqual(calc_win_state_bucket(0.40), 1)  # Weak
        self.assertEqual(calc_win_state_bucket(0.55), 2)  # Medium
        self.assertEqual(calc_win_state_bucket(0.75), 3)  # Strong

    def test_is_paired(self):
        cards = [Card.new("As"), Card.new("Ah")]
        self.assertTrue(is_paired(cards))

        cards = [Card.new("As"), Card.new("Kd")]
        self.assertFalse(is_paired(cards))

    def test_is_board_connected(self):
        cards = [Card.new("7d"), Card.new("8h"), Card.new("9s")]
        self.assertTrue(is_board_connected(cards))

        cards = [Card.new("7d"), Card.new("Kd"), Card.new("2c")]
        self.assertFalse(is_board_connected(cards))

    def test_get_board_high_card_level(self):
        cards = [Card.new("2d"), Card.new("6h"), Card.new("9s")]
        self.assertEqual(get_board_high_card_level(cards), 1)  # High card: 9

        cards = [Card.new("Td"), Card.new("Jh"), Card.new("Qs")]
        self.assertEqual(get_board_high_card_level(cards), 2)  # High card: Q

    def test_count_high_cards(self):
        cards = [Card.new("Td"), Card.new("Jh"), Card.new("2s")]
        self.assertEqual(count_high_cards(cards), 1)

        cards = [Card.new("2d"), Card.new("3h"), Card.new("4s")]
        self.assertEqual(count_high_cards(cards), 0)

    def test_get_board_state_features(self):
        cards = [Card.new("Td"), Card.new("Jh"), Card.new("Qs")]
        features = get_board_state_features(cards)
        self.assertEqual(features["Paired Board"], 0)
        self.assertEqual(features["Connected Board"], 1)
        self.assertEqual(features["Board High Card"], 2)
        self.assertEqual(features["Count High Cards"], 2)
    
    def test_get_board_state_features2(self):
        cards = [Card.new("2h"), Card.new("3c"), Card.new("4s")]
        features = get_board_state_features(cards)
        self.assertEqual(features["Paired Board"], 0)
        self.assertEqual(features["Connected Board"], 1)
        self.assertEqual(features["Board High Card"], 0)
        self.assertEqual(features["Count High Cards"], 0)

    def test_get_board_state_features3(self):
        cards = [Card.new("2h"), Card.new("3c"), Card.new("2s")]
        features = get_board_state_features(cards)
        self.assertEqual(features["Paired Board"], 1)
        self.assertEqual(features["Connected Board"], 0)
        self.assertEqual(features["Board High Card"], 0)
        self.assertEqual(features["Count High Cards"], 0)

    def test_get_rank_of_pair(self):
        player_cards = [Card.new("Td"), Card.new("Th")]
        board_cards = [Card.new("2d"), Card.new("5h"), Card.new("9s")]
        self.assertEqual(get_rank_of_pair(player_cards, board_cards), 2)  # Top pair

    def test_get_number_of_paired_cards(self):
        player_cards = [Card.new("Td"), Card.new("Th")]
        board_cards = [Card.new("2d"), Card.new("5h"), Card.new("9s")]
        self.assertEqual(get_number_of_paired_cards(player_cards, board_cards), 1)

    def test_get_flush_status(self):
        player_cards = [Card.new("2h"), Card.new("3h")]
        board_cards = [Card.new("4h"), Card.new("5h"), Card.new("6h")]
        self.assertEqual(get_flush_status(player_cards, board_cards), 2)  # Flush complete

    def test_get_straight_status(self):
        player_cards = [Card.new("2h"), Card.new("3d")]
        board_cards = [Card.new("4s"), Card.new("5h"), Card.new("6c")]
        self.assertEqual(get_straight_status(player_cards, board_cards), 2)  # Straight complete

    def test_get_full_house_status(self):
        player_cards = [Card.new("2h"), Card.new("2d")]
        board_cards = [Card.new("2s"), Card.new("5h"), Card.new("5c")]
        self.assertEqual(get_full_house_status(player_cards, board_cards), 1)  # Full house

    def test_get_player_board_features(self):
        player_cards = [Card.new("Td"), Card.new("Th")]
        board_cards = [Card.new("2d"), Card.new("5h"), Card.new("9s")]
        features = get_player_board_features(player_cards, board_cards)
        self.assertEqual(features["Rank of Pair"], 2)
        self.assertEqual(features["Number of Paired Cards"], 1)
        self.assertEqual(features["Flush"], 0)
        self.assertEqual(features["Straight"], 0)
        self.assertEqual(features["Full House"], 0)

if __name__ == "__main__":
    unittest.main()
