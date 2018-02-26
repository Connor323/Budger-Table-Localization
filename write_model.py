import yaml
import numpy as np 
import lxml.etree as ET

params_file = open("budger_localizer_params.yaml", "r")
params = yaml.load(params_file)

# create the file structure
model = ET.Element('model')  
for i in range(params["Localizer"]["budger_shape"][0]):
	for j in range(params["Localizer"]["budger_shape"][1]):
		item = ET.SubElement(model, 'point')  
		item.set('ID','(%d, %d)' % (i, j))  
		item.set('Coordinates','(%f, %f, %f)' % (i*params["Localizer"]["budger_distance_3d"], j*params["Localizer"]["budger_distance_3d"], 0))  

# create a new XML file with the results
mydata = ET.tostring(model, pretty_print=True)  
myfile = open(params["Localizer"]["model_file"], "w")  
myfile.write(mydata)  