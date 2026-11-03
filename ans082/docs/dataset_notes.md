# Dataset Notes

## Main dataset
- **Name:** Garbage Classification (12 classes)
- **Source:** https://www.kaggle.com/datasets/mostafaabla/garbage-classification
- **Owner / author:** <from the Kaggle page>
- **Version / date:** <from the Kaggle page>
- **Licence:** <exact text from the Kaggle page>
- **Size:** about 250 MB, 15,515 images in 12 class folders
- **Original classes and counts:** battery 945, biological 985, brown-glass 607, cardboard 891, clothes 5325, green-glass 629, metal 769, paper 1050, plastic 865, shoes 1977, trash 697, white-glass 775

## Class merge
brown-glass, green-glass and white-glass were merged into one class `glass` (2,011 images),
giving 10 classes. Glass colour does not change which bin the item goes in, so this removes
a distinction the application does not need. Raw folders are left untouched; the merge is
done by label mapping when the split file is built.

## Known issues
- Strong class imbalance: clothes (5,325) is about 7.6 times larger than trash (697).
- <add anything else you notice later>

## Related dataset (not trained on)
- TrashNet (Yang and Thung, Stanford, 2016): about 2,500 images, 6 classes
  (glass, paper, cardboard, plastic, metal, trash). Discussed in Related Work.
  Source URL and licence: <check and fill in>

## Real-photo test set
- <total> photos taken with my own phone (<phone model>), <min>-<max> per class, 10 class folders.
- Varied backgrounds, lighting, angles, and at least 5 different items per class.
- Photos contain no faces or personal information; location metadata was removed.
- Resized to a maximum of 1024 px on the long side with `src/resize_photos.py`.
- Stored on Google Drive (`SmartSort_AI/real_photos`), not in Git.
- Never used for training or tuning; used once at the end to measure the real-world gap.