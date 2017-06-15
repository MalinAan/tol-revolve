import sys
import os
from string import Template
import math

from sdfbuilder import Link, Model, SDF
from sdfbuilder.math import Vector3
from sdfbuilder.structure import Box, Cylinder, Collision, Visual, StructureCombination

#template for model config files
config_string=Template("""<?xml version="1.0"?>
<model>
  <name>${model_name}</name>
  <version>1.0</version>
  <sdf version='1.5'>model.sdf</sdf>
  <author>
   <name>Kyrre Glette</name>
   <email>kyrrehg@ifi.uio.no</email>
  </author>
  <description>
    Generated environment model
  </description>
</model>
""")

#writes an sdfbuilder sdf and config file to have a gazebo model
def write_model(model_name,sdf):
    #make models directory if not exitsting
    models_path="models"
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    #make model directory
    model_path=os.path.join(models_path,model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    #write config file
    conf_str=config_string.safe_substitute({'model_name' : model_name})
    config_file_name=os.path.join(model_path,"model.config")
    #if not os.path.isfile(config_file_name):
    with open(config_file_name, "w") as config_file:
        print("writing conf file")
        config_file.write(conf_str)

    model_file_name=os.path.join(model_path,"model.sdf")
    with open(model_file_name, "w") as sdf_file:
        sdf_file.write(str(sdf))


#creates an sdfbuilder model with a grid of boxes
def gen_boxes(model_name, dimensions=4, spacing=0.5, size=0.05):
    model = Model(model_name,static=True)
    for x in range(-dimensions,dimensions+1):
        for y in range(-dimensions,dimensions+1):
            l = Link("box")

            #box_geom = Box(1.0, 1.0, 1.0, mass=0.5)
            #b = StructureCombination("box", box_geom)
            #l.add_element(b)

            #height=0.005+0.007*max(abs(x),abs(y))
            height=0.01
            center_sq=3
            if (abs(x)>=center_sq or abs(y)>=center_sq):
                l.make_box(1, size, size, height)

            pos = Vector3(spacing*x,spacing*y,height/2)

            l.set_position(pos)
            #l.rotate_around(Vector3(0, 0, 1), math.radians(x*y), relative_to_child=False)

            model.add_element(l)
    return model



#creates an sdfbuilder model with a grid of boxes
def gen_steps(model_name, numsteps=4, stepheight=0.01, size=0.1, incline=0.0):
    model = Model(model_name,static=True)
    for x in range(0,numsteps):
            l = Link("box")
            height=0.01
            offset=0.4
            l.make_box(1, size, size, stepheight)
            pos = Vector3(offset+size*x,0,stepheight/2+x*stepheight)
            l.set_position(pos)
            #l.rotate_around(Vector3(0, 0, 1), math.radians(x*y), relative_to_child=False)
            model.add_element(l)
    return model



#generate a grid of boxes and write to file
sdf = SDF()
model_name="boxes_grid"
model=gen_boxes(model_name, dimensions=8,spacing=0.15, size=0.04)
sdf.add_element(model)
write_model(model_name,sdf)

#generate steps
sdf = SDF()
model_name="steps_exp"
model=gen_steps(model_name)
sdf.add_element(model)
write_model(model_name,sdf)
