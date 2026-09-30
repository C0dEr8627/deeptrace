# Project Plan

## 1. Objective

Deliver a functional semester prototype for visual deepfake screening of
images and videos while keeping implementation scope and compute
requirements manageable.

## 2. Phased schedule

  -----------------------------------------------------------------------
  Phase                   Activities              Deliverable
  ----------------------- ----------------------- -----------------------
  1\. Requirements        Confirm scope, file     Approved SRS and design
                          limits, user flow, and  
                          evaluation approach     

  2\. Dataset preparation Select accessible       Dataset manifest and
                          dataset, inspect        split
                          labels, create          
                          source-aware splits     

  3\. Image model         Fine-tune pretrained    Working image inference
                          classifier and evaluate 
                          checkpoint              

  4\. Video pipeline      Sample frames, run      Working video inference
                          image model, aggregate  
                          scores                  

  5\. API                 Implement upload        FastAPI service
                          validation, processing  
                          orchestration, response 
                          schema                  

  6\. Frontend            Build upload, progress, Usable web interface
                          error, and result       
                          screens                 

  7\. Integration         Connect frontend, API,  End-to-end prototype
                          and model; fix issues   

  8\. Evaluation and      Run test plan, document Final demonstration and
  report                  metrics and limitations report
  -----------------------------------------------------------------------

The schedule can be compressed or expanded to match the semester
calendar. Complete a basic image workflow before investing time in video
support.

## 3. Minimum viable deliverable

-   One fine-tuned pretrained image classifier.
-   Image upload and inference.
-   Video frame sampling and frame-score aggregation.
-   Simple web interface.
-   Held-out evaluation with appropriate metrics.
-   Documentation of uncertainty, limitations, and test conditions.

## 4. Optional extensions

Only after the MVP is stable: - Face detection/cropping if it improves
validation results. - Frame-level evidence visualization. - Additional
dataset for cross-dataset testing. - Explainability visualization such
as Grad-CAM, with a note that attribution is not proof. - Audio or
temporal models as future work, not a dependency for completion.

## 5. Risks

  -----------------------------------------------------------------------
  Risk                                Mitigation
  ----------------------------------- -----------------------------------
  Limited GPU or time                 Transfer learning, small input
                                      size, bounded experiments

  Dataset access/licensing            Verify terms before downloading or
                                      redistributing

  Poor generalization                 Source-aware split and honest
                                      cross-condition evaluation

  Video processing too slow           Limit duration and sampled frames

  Misleading output                   Use calibrated language,
                                      uncertainty band, and limitations

  Integration delays                  Validate model inference
                                      independently before API/UI
                                      integration
  -----------------------------------------------------------------------

## 6. Completion checklist

-   [ ] Requirements and scope confirmed
-   [ ] Dataset source and terms recorded
-   [ ] Data split avoids source leakage where possible
-   [ ] Image model trained and evaluated
-   [ ] Video sampling and aggregation tested
-   [ ] API validation and error handling tested
-   [ ] Frontend connected to API
-   [ ] Metrics and test evidence recorded
-   [ ] Known limitations documented
-   [ ] Final demo and report prepared
