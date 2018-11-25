Logistic Regression README!

In order to run the code properly one must first of all change the file directory in lines 103-104

Once that's done, all functions should be run (from line 1 to line 102).
To train the model, run lines 103-161.
The default number of features used is 4000, but can be modified (line 145)
The default feature type is TF-IDF but can be modified to BoW by commenting out line 151, and commenting in line 152.

Lines 176-295 Run the 5-fold cross validation.

Expect a runtime of up to 600 seconds (depends on machine). If the number of features is increased, runtime can increase up to 2 hours for one round of k-fold.