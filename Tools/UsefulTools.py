


# output ranking list	
def outputRank(query_docs_point_dict):
	cquery_docs_point_dict = sorted(query_docs_point_dict.items(), key=operator.itemgetter(0))
	operation = "w"
	with codecs.open("Query_Results.txt", operation, "utf-8") as outfile:
		for query, docs_point_list in query_docs_point_dict.items():
			outfile.write(query + "\n")	
			out_str = ""
			for docname, score in docs_point_list:
				out_str += docname + " " + str(score) + "\n"
			outfile.write(out_str)
			outfile.write("\n")			
			
# merge two dict			
def mergeTwoDicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z			