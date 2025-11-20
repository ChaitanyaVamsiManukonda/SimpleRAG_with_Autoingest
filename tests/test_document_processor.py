import sys
import os
import unittest
import shutil

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.document_processor import DocumentProcessor

class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        # Use a smaller model for speed in tests if possible, or just the default
        # We'll use the default but maybe mock it if it's too slow, but for now let's run it real.
        self.processor = DocumentProcessor()

    def test_chunking_logic(self):
        text = "This is sentence one. " * 10 + "This is sentence two. " * 10
        # Make it long enough to split
        long_text = text * 5
        
        chunks = self.processor.create_document_chunks(long_text)
        self.assertTrue(len(chunks) > 0)
        
        # Check overlap (basic check)
        if len(chunks) > 1:
            # This is a loose check because exact overlap depends on tokenization
            self.assertTrue(len(chunks[0]['text']) > 0)
            self.assertTrue(len(chunks[1]['text']) > 0)

    def test_small_chunk_logic(self):
        # This specifically targets the "pass" block in the original code
        # We want to see if it handles small leftovers gracefully
        text = "Short sentence."
        chunks = self.processor.create_document_chunks(text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]['text'], "Short sentence.")

    def test_empty_document(self):
        chunks = self.processor.create_document_chunks("")
        self.assertEqual(len(chunks), 0)

if __name__ == '__main__':
    unittest.main()
