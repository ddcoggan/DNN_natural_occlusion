# Downloaded model weights and distribute into the appropriate directories
# The download is a zip file of size 27.3GB

import os.path as op
import zipfile
import requests
from io import StringIO

zip_file_url = 'https://drive.google.com/file/d/1awH0JBmTvz2yARLwnuBEQntySqyh-XT5/view?usp=sharing'
r = requests.get(zip_file_url, stream=True)
z = zipfile.ZipFile(StringIO.StringIO(r.content))

""" Alternative to the above code: download the zip file from the link above 
and place it in the top-level directory. Then run the line below to create the 
zipfile object. """
# z = zipfile.ZipFile('DNN_natural_occlusion_weights.zip')

for member in z.infolist():
    if member.filename.endswith('/weights.pt'):
        target_path = member.filename.replace('weights/', 'DNN/models/')
        print(target_path)
        if not op.isfile(target_path):
            with open(target_path, 'wb') as f:
                f.write(z.read(member.filename))