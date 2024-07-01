#!/usr/bin/env python3

# calls the PDS converter tool on all 2.0 glTF models

import os, sys

# delete the converted directory
os.system("rm -rf converted")

# walk all directories in gltf-sample-models/2.0 , then for
#   each directory that has a glTF directory,
#   make a new directory under converted/dirname
#   call the PDS converter tool on the glTF file and output in the directory,
#   and copy all the bin files to the directory
def convertModels():
  root = "gltf-sample-models/2.0"
  for dirs in next(os.walk("gltf-sample-models/2.0")):
    if (dirs == "gltf-sample-models/2.0"):
      continue
    for dir in dirs:
      # check if dir is a directory
      if not os.path.isdir(f"{root}/{dir}"):
        continue
      # check if dir has a glTF directory
      if not os.path.isdir(f"{root}/{dir}/glTF"):
        print(f"ERR: no glTF directory in {dir}")
        continue
      # check if the glTF file exists
      if not os.path.isfile(f"{root}/{dir}/glTF/{dir}.gltf"):
        print(f"ERR: no glTF file in {dir}")
        continue
      # check directory has spaces as not supported yet
      if " " in dir:
        print(f"ERR: directory name {dir} has a space, which is not supported")
        continue
      os.makedirs(f"converted/{dir}", exist_ok=True)
      # the glTF file will match the directory name
      cmd = (
        f"puledit json-convert"
        f" --src {root}/{dir}/glTF/{dir}.gltf"
        f" --dst converted/{dir}/{dir}.pds"
      )
      os.system(cmd)
      # copy all bin and png files to the directory, ignore warnings
      os.system(f"cp {root}/{dir}/glTF/*.bin converted/{dir}/ 2>/dev/null")
      os.system(f"cp {root}/{dir}/glTF/*.png converted/{dir}/ 2>/dev/null")
      # convert jpg to png
      #os.system(f"cp {root}/{dir}/glTF/*.jpg converted/{dir}/ 2>/dev/null")
      #os.system(f"mogrify -format png converted/{dir}/*.jpg")

convertModels()
