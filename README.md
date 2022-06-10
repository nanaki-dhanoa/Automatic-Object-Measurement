# Automatic-Object-Measurement
## Calculates the size of objects based on a given reference object
This object size estimator is created using opencv and numpy.
The website is created using streamlit.

# How to run this project?
1. Download the repository and unzip it
2. Open the command prompt from within this folder and create a virtual env to install the project dependencies.
  (You can do this using the following command for Windows OS)
  ```
  > python -m venv {venv-name}
  ({venv-name}) > {venv-name}\Scripts\activate
  ({venv-name}) > python -m pip install --upgrade pip
  ({venv-name}) > python -m pip install -r requirements.txt
  ```
3. Once all the dependencies are successfully installed, you can run the app on your browser using the command
  ```
  ({venv-name}) > streamlit run app.py
  ```
  The webpage should look something like this
  
  ![alt text](https://github.com/nanaki-dhanoa/Automatic-Object-Measurement/blob/main/readme_images/image1.png?raw=true)
  
  Upload the image you want to perform measurement on (*there should be a reference square of 2x2 cm as the leftmost element in the image*) and then click on `Measure Distance` / `Measure Area` button
  
  ![alt text](https://github.com/nanaki-dhanoa/Automatic-Object-Measurement/blob/main/readme_images/image2.png?raw=true)
  
  ![alt text](https://github.com/nanaki-dhanoa/Automatic-Object-Measurement/blob/main/readme_images/image3.png?raw=true)
  
You can also download the output images using the `Download` button
