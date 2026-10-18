Task: classify 96 by 96 colour images into cardboard, glass, metal, paper, plastic and trash with a small network. Limits: ONNX file, at most 350,000 parameters, at most 1.5 MB, at least 75% on a hidden test set.

Final model: a 3 layer convolution network with 5 by 5 windows (32 filters each), max pooling, global average pooling and a final layer, trained with augmentation. 53,894 parameters, 0.21 MB, about 1 ms per image on CPU.

Results: clean validation 100% (240 images), clean train 99.78% (900 images). Average of three runs of the final design on altered validation images: darker 94.3%, colour 0.3 95.3%, colour 0.05 72.2%, noise 93.8%, blur 84.3%, rotate 30 87.8%.

Repo contents: model/model.onnx is the submission. scripts holds the training and checking code. experiment_log.txt records every experiment. learning_log.txt records what I learned each day.

Data: 900 training images (150 per class) and 240 validation images (40 per class). The images look computer generated. Cardboard is brown with varied shapes, paper is mostly white with coloured lines, metal is round with a small white or grey rectangle near the centre, trash is random zigzag patterns with a white circle, glass has a rectangular body with a square neck and a white vertical line on the left, and plastic has two colours one for the body and one for the cap, with horizontal lines on a squarish body. Glass and plastic looked most alike. Classes are balanced.

What I did: the starter model scored 100% on validation, so validation could not rank models. and i ran a stress test that alters validation images and found that darker images were all called glass. Then I changed one thing at a time and trained each version 3 times.

Experiments: 
    0: starting model, kept as baseline
    1: augmentation, kept.
    2: window size 3, dropped.
    3: pooling by 2, dropped.
    4: third layer, kept.
    5: wider colour range, kept.
Details and numbers are in experiment_log.txt.

What did not work: 3 by 3 windows and shrinking by 2 both lowered accuracy, and both lowered training accuracy too, which means they underfit: they learned the task worse.

How to reproduce: put the starter kit data in data/public/train and data/public/val. Run: python scripts/train_exp5.py --epochs 30 --batch_size 32 --lr 0.001 --output model.onnx
Check it with: python scripts/verify_submission.py --model model.onnx --val_dir data/public/val
Results differ slightly between runs. train_exp5.py comes from train_template.py through small edits (augmentation in train_exp1.py, third layer in train_exp4.py, colour range in train_exp5.py). Experiment 2 changed kernel_size=5 to 3 and Experiment 3 changed MaxPool2d(4, 4) to MaxPool2d(2, 2).

Use of AI assistance: I used an AI assistant (Claude) throughout the project for guidance. It explained concepts. I suggested the experiments and what all can be done and altered, since I had no prior knowledge about about ML and pytorch library I used claude's assistance to write most of the codes (the stress test, augmentation, the third layer edit, comparison scripts). I ran every experiment, chose which ones to keep and all the decisions and alteration ideas were done by me. I had also used AI's help to edit and alter the logs. and it is to be noted that i am a total beginner to this field and had only been studying and working on it for around 10 days approx.

Repo: https://github.com/Ram08-gh/Int_SIG-REC
