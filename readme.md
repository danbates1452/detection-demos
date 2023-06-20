# Detection Demos

At the time of writing, YOLOv8 is the most performant and supports multiple people in frame.

Note: For Mediapipe demos, models will not self-install, see the relevant solution's guide page to download: https://developers.google.com/mediapipe/solutions/guide

Install Steps:
1. Create Python3.8 Virtual Environment: ``python3.8 -m venv venv``
2. Activate Virtual Environment: ``.\venv\Scripts\activate``
3. Install Torch with Cuda: ``pip3.8 install torch torchvision --index-url https://download.pytorch.org/whl/cu117`` 
4. Install All Other Requirements: ``pip3.8 install -r .\requirements.txt``

Note: You may have to uninstall torch and reinstall it with cuda again with the command above as it doesn't always initially pull cuda with it.