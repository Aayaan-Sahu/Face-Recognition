# Face Detection

## General
- When making your training data, make sure to choose a good number of samples. Something like 400-800 is good.
- Make sure that there is an equal number of samples for each class (or person that the model will try to recognize).

## Usage
1. Run the `create_data.py` file to create training data.  
- Sample usage is: `python3 create_data.py --person="{person_name}" --samples="{number_of_samples}"`
- This will create a folder named `{person_name}` in the current directory.
- Do this for each person you want to train the model to recognize.

2. Create a `data` folder in the current directory.

3. Move the folders containing the training data to the `data` folder.

4. Run the `learn.py` file.
* You can run this file on your computer but you might not have a capable enough GPU to run the script at reasonable speeds. Thus, you can use the service `Google Colaboratory` to run the script. Refer to the `Google Colab` section for more information.

5. Run the `detect_image.py` file.
- Sample usage is: `python3 detect_image.py --path {image_path}`

6. Run the `detect_video.py` file.
- Sample usage is: `python3 detect_video.py --person="{person_name}"`

## Google Colab
To run the script on Google Colab, first create a folder in your Google Drive where you will create the script and house the training data. Next create a folder in your Google drive called `data` and move the folders containing the training data to this folder. It'll take some time to upload, but from there you are ready to run the `learn.py` script. In the main directory within Google Colab, create a new Google Colaboratory notebook and paste the `learn.py` script into it. Comment out the section that says `# Google Colab` and comment the section that says `# Local`. Change your runtime type to **GPU** and run the script. This will produce two files in the Google Colab directory called `detector` and `labels.json`. Download these two files into the local directory where you created the training data, and you are set to run the `detect_image.py` script.