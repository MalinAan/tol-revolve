import sys
import os
from string import Template
import math

from sdfbuilder import Link, Model, SDF, PosableGroup
from sdfbuilder.math import Vector3, Quaternion
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
def gen_boxes(model_name, dimensions=4, spacing=0.5, size=0.05, height=0.015, center_sq=1):
    model = Model(model_name,static=True)
    for x in range(-dimensions,dimensions+1):
        for y in range(-dimensions,dimensions+1):
            l = Link("box")
            if (abs(x)>=center_sq or abs(y)>=center_sq):
                l.make_box(1, size, size, height)
            pos = Vector3(spacing*x,spacing*y,height/2)
            l.set_position(pos)
            #l.rotate_around(Vector3(0, 0, 1), math.radians(x*y), relative_to_child=False)
            model.add_element(l)
    return model



#steps
def gen_steps(model_name, num_steps=6, offset=0.4, height=0.02, width=3.0, depth=0.2, incline=0):
    model = Model(model_name,static=True)
    steps = PosableGroup()
    for x in range(0,num_steps):
        l = Link("box")
        l.make_box(1, depth, width, height)
        pos = Vector3(offset+depth*x,0,height/2+x*height)
        l.set_position(pos)
        steps.add_element(l)

    for x in range(0,4):
        steps.set_rotation(Quaternion.from_rpy(0,math.radians(-incline),math.radians(90*x)))
        model.add_element(steps.copy())

    return model



#generate a grid of boxes and write to file
sdf = SDF()
model_name="boxes_grid"
model=gen_boxes(model_name, dimensions=6,spacing=0.3, size=0.04, height=0.02)
sdf.add_element(model)
write_model(model_name,sdf)

#generate steps
sdf = SDF()
model_name="exp2_steps"
model=gen_steps(model_name, incline=4, offset=0.8)
#model.set_position(Vector3(0,0,-0.041)) #inaccurate height compensation due to the incline - does not seem to affect the model?
sdf.add_element(model)
write_model(model_name,sdf)

#different boxes
sdf = SDF()
model_name="exp2_boxes"
model=gen_boxes(model_name, dimensions=6,spacing=0.3, size=0.08, height=0.04, center_sq=0)
sdf.add_element(model)
write_model(model_name,sdf)

#boxes which act as platforms, chasms between
sdf = SDF()
model_name="exp2_chasms"
model=gen_boxes(model_name, dimensions=6,spacing=0.4, size=0.3, height=0.04, center_sq=0)
sdf.add_element(model)
write_model(model_name,sdf)

#different boxes
sdf = SDF()
model_name="exp2_pillars"
model=gen_boxes(model_name, dimensions=4,spacing=0.3, size=0.04, height=0.2, center_sq=2)
sdf.add_element(model)
write_model(model_name,sdf)
