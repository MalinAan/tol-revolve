from __future__ import print_function
import sys
from sdfbuilder import Link, Model, SDF
from sdfbuilder.math import Vector3
from sdfbuilder.structure import Box, Cylinder, Collision, Visual, StructureCombination

import math

sdf = SDF()
model = Model("obstacles",static=True)


def gen_boxes(dimensions=4, spacing=0.5, size=0.05):
        for x in range(-dimensions,dimensions+1):
            for y in range(-dimensions,dimensions+1):
                l = Link("box")

                #box_geom = Box(1.0, 1.0, 1.0, mass=0.5)
                #b = StructureCombination("box", box_geom)
                #l.add_element(b)

                if (x!=0 or y!=0):
                    l.make_box(1, size, size, 0.01*max(abs(x),abs(y)))

                pos = Vector3(spacing*x,spacing*y,0)

                l.set_position(pos)
                l.rotate_around(Vector3(0, 0, 1), math.radians(x*y), relative_to_child=False)

                model.add_element(l)

#adjust it up to ground plane
model.set_position(Vector3(0, 0, 0.0))

sdf.add_element(model)

gen_boxes(dimensions=4,spacing=0.15, size=0.03)
print(str(sdf))
