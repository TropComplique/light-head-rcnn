## WIDER

1. sudo apt-get install octave
2. http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/eval_script/eval_tools.zip
3.  ```
    wider_eval.m
    pred_dir = '../result';
    octave wider_eval.m
    ```
4.  ```
    error: save: Unrecognized option '-v7.3'
    error: called from
        evaluation at line 48 column 1
        wider_eval.m at line 26 column 5

    save(sprintf('./plot/baselines/Val/%s/%s/wider_pr_info_%s_%s.mat',setting_class,legend_name,legend_name,setting_name),'pr_cruve','legend_name','-v7.3');
    ```

# FDDB


## How to evaluate on FDDB

1. Download the evaluation code from [here](http://vis-www.cs.umass.edu/fddb/results.html).
2. `tar -zxvf evaluation.tgz; cd evaluation`.
Then compile it using `make` (it can be very tricky to make it work).
3. Run `predict_for_FDDB.ipynb` to make predictions on the evaluation dataset.
You will get `ellipseList.txt`, `faceList.txt`, `detections.txt`, and `images/`.
4. Run `./evaluate -a result/ellipseList.txt -d result/detections.txt -i result/images/ -l result/faceList.txt -z .jpg -f 0`.
5. You will get something like `eval_results/discrete-ROC.txt`.
6. Run `eval_results/plot_roc.ipynb` to plot the curve.

Also see this [repository](https://github.com/pkdogcom/fddb-evaluate) and the official [FAQ](http://vis-www.cs.umass.edu/fddb/faq.html) if you have questions about the evaluation.
