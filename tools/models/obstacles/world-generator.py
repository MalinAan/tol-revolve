from __future__ import print_function
import sys
from sdfbuilder import Link, Model, SDF
from sdfbuilder.math import Vector3
from sdfbuilder.structure import Box, Cylinder, Collision, Visual, StructureCombination

import math

sdf = SDF()
model = Model("obstacles")

for y in range(-5,5):
    for x in range(-5,5):
        l = Link("box")

        #box_geom = Box(1.0, 1.0, 1.0, mass=0.5)
        #b = StructureCombination("box", box_geom)
        #l.add_element(b)

        l.make_box(1, 0.1, 0.1, 0.01+0.01*abs(x))

        pos = Vector3(0.6*x,0.6*y,0)

        l.set_position(pos)
        l.rotate_around(Vector3(0, 0, 1), math.radians(x*4), relative_to_child=False)

        model.add_element(l)

#adjust it up to ground plane
model.set_position(Vector3(0, 0, 0.0))

sdf.add_element(model)


print(str(sdf))
