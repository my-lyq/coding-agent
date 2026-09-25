import json
import tempfile
import unittest
from pathlib import Path
import torch
from dataset import CausalLMCollator, TrajectorySFTDataset, expand_trajectory

class FakeTokenizer:
    pad_token_id = 0
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = "<user>" + messages[0]["content"] + "<assistant>"
        if len(messages) == 2:
            text += messages[1]["content"] + "<eos>"
        return list(text.encode())

class DatasetTest(unittest.TestCase):
    def setUp(self):
        self.record={"instruction":"Fix bug","input":"a.py: bad","trajectory":[{"thought":"inspect","action":"read_file({})","observation":"bad"},{"thought":"test","action":"run_test({})","observation":"OK"}],"answer":"--- a/a.py\n+++ b/a.py\n-bad\n+good"}
    def test_expands_steps_and_final_patch(self):
        examples=expand_trajectory(self.record)
        self.assertEqual(len(examples),3)
        self.assertNotIn("Observation: bad",examples[0]["prompt"])
        self.assertIn("Observation: bad",examples[1]["prompt"])
        self.assertTrue(examples[-1]["target"].startswith("Final Answer:"))
    def test_masks_prompt_and_collates(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"train.jsonl"; path.write_text(json.dumps(self.record)+"\n")
            dataset=TrajectorySFTDataset(path,FakeTokenizer(),max_length=4096)
            item=dataset[0]
            self.assertIn(-100,item["labels"])
            self.assertTrue(any(label!=-100 for label in item["labels"]))
            batch=CausalLMCollator(0)([item,dataset[1]])
            self.assertEqual(set(batch),{"input_ids","attention_mask","labels"})
            self.assertIsInstance(batch["input_ids"],torch.Tensor)

if __name__=="__main__": unittest.main()
