"""Synthetic preprocessing checks; these are not the required collected digits."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image,ImageDraw

from digits import preprocess_digit,load_own_digits


class DigitTests(unittest.TestCase):
    def test_polarity_shape_and_read_only_source(self):
        with tempfile.TemporaryDirectory() as directory:
            arrays=[]
            for background,ink in [(0,255),(255,0)]:
                p=Path(directory)/str(background);p=p.with_suffix('.png')
                image=Image.new('L',(80,60),background)
                ImageDraw.Draw(image).line([(30,10),(30,50)],fill=ink,width=7)
                image.save(p);original=p.read_bytes()
                result=preprocess_digit(p);arrays.append(result)
                self.assertEqual(result.shape,(28,28,1));self.assertEqual(result.dtype,np.float32)
                self.assertGreater(result.max(),.9);self.assertEqual(p.read_bytes(),original)
            np.testing.assert_allclose(arrays[0],arrays[1],atol=.01)

    def test_blank_is_rejected_and_missing_samples_remain_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'blank.png';Image.new('L',(28,28),255).save(p)
            with self.assertRaises(ValueError):preprocess_digit(p)
            images,info=load_own_digits(directory)
            self.assertIsNone(images);self.assertEqual(info['missing_digits'],list(range(10)))

    def test_transparent_bright_ink_has_dark_background(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'digit.png';image=Image.new('RGBA',(60,60),(0,0,0,0))
            ImageDraw.Draw(image).line([(30,10),(30,50)],fill='white',width=7);image.save(p)
            result=preprocess_digit(p)
            self.assertGreater(result.max(),.9);self.assertEqual(float(result[0].sum()),0)


if __name__=='__main__':unittest.main()
