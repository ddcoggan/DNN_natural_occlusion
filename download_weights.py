# Downloaded model weights and distribute into the appropriate directories
# The download is a zip file of size 27.3GB

import os.path as op
import zipfile
import requests
from io import StringIO

zip_file_url = 'DNN_natural_occlusion_weights.zip'
r = requests.get(zip_file_url, stream=True)
z = zipfile.ZipFile(StringIO.StringIO(r.content))

for member in z.infolist():
    if member.filename.endswith('.pt'):
        target_path = op.join('DNN/models', member.filename[:-3], 'weights.pt')
    with open(target_path, 'wb') as f:
        f.write(z.read(member.filename))