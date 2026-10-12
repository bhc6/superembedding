import unittest
import run


class ScoringTests(unittest.TestCase):
    def test_no_answer_extraction_from_reasoning(self):
        for text in ['Yes',' yes.\n','NO!']:self.assertIn(run.parse_answer(text),('yes','no'))
        for text in ['Yes, because a cat is visible.','<think>No</think>Yes','Yesterday','Yes No','']:
            self.assertEqual(run.parse_answer(text),'invalid')

    def test_invalid_counts_wrong_in_both_metrics(self):
        rows=[{'id':'1','actual':'yes','pred':'yes'},
              {'id':'2','actual':'yes','pred':'invalid'},
              {'id':'3','actual':'no','pred':'invalid'}]
        m=run.metrics(rows,'pred')
        self.assertEqual(m['accuracy'],1/3)
        self.assertEqual(m['balanced_accuracy'],.25)
        self.assertEqual(m['invalid'],2)
        with self.assertRaises(ValueError):run.metrics(rows+rows[:1],'pred')


if __name__=='__main__':unittest.main()
