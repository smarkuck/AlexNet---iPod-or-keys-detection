# AlexNet - iPod or keys detection
Simple mobile application which detects iPod and keys in real time.

<p float="left">
  <img src="https://github.com/smarkuck/AlexNet---iPod-or-keys-detection/blob/master/images/ipod.jpg" alt="drawing" width="300"/>
  <img src="https://github.com/smarkuck/AlexNet---iPod-or-keys-detection/blob/master/images/keys.jpg" alt="drawing" width="300"/>
  <img src="https://github.com/smarkuck/AlexNet---iPod-or-keys-detection/blob/master/images/nothing.jpg" alt="drawing" width="300"/>
</p>

**data.hdf5** - 3000 training images with size 227x227</br>
**test.hdf5** - 750 test images with size 227x227

## How to use
1. Download training data + trained model from here: https://drive.google.com/file/d/1zxZI0Io_uDfWrQ635cRshWIR7H2A9sHS/view?usp=sharing
2. To train own model, open Jupiter notebook in Google Colaboratory: **AlexNetGenerator.ipynb**
3. In first cell, set proper Google Drive paths for training data and model output
4. Execute cells step by step to train and save own model
5. Replace **mobileApp/app/src/main/assets/model.tflite** with downloaded or trained model
6. Open **mobileApp** project in Android Studio
7. Build mobile application
