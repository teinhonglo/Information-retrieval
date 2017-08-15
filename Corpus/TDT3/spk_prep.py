import os


def read_file(filepath):
	data = {}				
	# content of document (doc, content)
	# list all files of a directory(Document)
	for dir_item in os.listdir(filepath):
		# join dir path and file name
		dir_item_path = os.path.join(filepath, dir_item)
		# check whether a file exists before read
		if os.path.isfile(dir_item_path):
			with open(dir_item_path, 'r') as f:
				# read content of document (doc, content)
				content = ""
				count = 0
				for line in f.readlines():
					WID = line.split()[0]
					content += WID + " "
					if WID == "-1" or count < 3:
						content += "\n"
					count += 1	
				data[dir_item]	= content
	return data	
	
def main():
	filepath = "SPLIT_AS0_WDID_NEW"
	txtpath = "SPLIT_DOC_WDID_NEW"
	data = read_file(filepath)
	txt_data = read_file(txtpath)
	doc_list = txt_data.keys()
	# remove duplicate document
	for filename in data.keys():
		if not filename in doc_list:
			data.pop(filename, None)
			
	for filename, content in data.items():
		with open(filepath + "_C/"+filename, "wb") as f:
			f.write(content)
	
if __name__ == "__main__":
	main()