#!/bin/bash
EXPERIMENT="$1."
EXPERIMENT_WITHOUT_DOT="$1"
INPUT=exp_makes.csv
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }

while IFS=, read -r dirname robotid rest || [[ -n "$dirname" ]]
do
	# Remove prefix and suffix quotes from dirname, which is <experiment name>.<run>
	dirname="${dirname%\"}"
	dirname="${dirname#\"}"

	# Check if experiment is prefix of the dirname
	if [[ $dirname =~ ^("$EXPERIMENT") ]]; then
		echo "Going to make ../../tools/models/$dirname"
		mkdir "../../tools/models/$dirname"

		echo "Copying robot_$robotid.sdf to ../../tools/models/$dirname/robot_$robotid.sdf"
		cp output/$EXPERIMENT_WITHOUT_DOT/robot_$robotid.sdf ../../tools/models/$dirname/robot_$robotid.sdf

		echo "Making model.config"
		echo "<?xml version="1.0"?>
		<model>
		  <name>$dirname</name>
		  <version>1.0</version>
		  <sdf version='1.5'>robot_$robotid.sdf</sdf>

		  <author>
		   <name>Malin Aandahl</name>
		   <email>mwaandah@uio.no</email>
		  </author>

		  <description>
		    Template for inspecting robots
		  </description>
		</model>" >> ../../tools/models/$dirname/model.config


		echo "Current files in ../../tools/models/$dirname/"
		echo $(ls "../../tools/models/$dirname/")
		echo ""

	fi;

done < $INPUT
