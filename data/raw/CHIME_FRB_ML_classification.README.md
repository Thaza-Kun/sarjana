#=============================================================
# Col_num, col_name, : contents/Explanations.
 001  tns_name       : Name ID provided for each burst. The same repeating FRB source has multiple tns_name due to its repetition.  A non-repeating FRB has the single tns_name. 
                            The same tns_name can have multiple sub-bursts.  Therefore, multiple rows with the same tns_name is due to the sub-bursts.
 002  classification : FRB classification provides by machine learning model.  1 = repeating FRB, 0 = repeating FRB candidate and -1 = non-repeating FRB
 003  group          : The clustering result for machine learning model output.  See Fig.3 of the paper. 
 004  embedding_x    : The x coordinate of the low dimensional projection of each FRB.  See Fig.2, 3, 4 of the paper.
 004  embedding_y    : The y coordinate of the low dimensional projection of each FRB.  See Fig.2, 3, 4 of the paper.