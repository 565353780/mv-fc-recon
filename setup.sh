cd ..
git clone https://github.com/NVIDIAGameWorks/kaolin.git

pip install ninja

pip3 install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

pip install scipy trimesh pickle pygltflib ipyevents \
  ipycanvas rtree warp-lang

cd kaolin
python setup.py install
