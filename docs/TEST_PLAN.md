# Test Plan

## 1. Purpose

Validate the application's core workflows, safe input handling, model
behavior, and reporting quality.

## 2. Test levels

### Unit tests

-   File validation accepts supported types and rejects unsupported
    types.
-   Image preprocessing returns the expected tensor shape and numeric
    range.
-   Video sampler respects duration and frame-count limits.
-   Inference returns a finite score in the expected range.
-   Aggregation produces the expected result for known sample scores.
-   Decision logic handles threshold boundaries and uncertain cases.

### Integration tests

-   Frontend can submit a valid image to the API.
-   Frontend can submit a valid video to the API.
-   API response follows the documented schema.
-   Invalid files return readable client errors.
-   Preprocessing, inference, aggregation, and cleanup work together.

### Model evaluation

Use held-out data that is separated by identity/source where metadata
permits. Avoid splitting near-duplicate frames or source videos across
training and test sets.

## 3. Functional test cases

  -----------------------------------------------------------------------
  ID                      Input/action            Expected result
  ----------------------- ----------------------- -----------------------
  TC-01                   Upload supported image  Analysis completes and
                                                  returns score/category

  TC-02                   Upload supported video  Bounded frames are
                                                  sampled and aggregated

  TC-03                   Upload unsupported      Request rejected with
                          extension               readable error

  TC-04                   Upload empty file       Request rejected

  TC-05                   Upload file above limit Request rejected before
                                                  expensive processing

  TC-06                   Corrupt image/video     Safe decoding error is
                                                  returned

  TC-07                   Video with too few      Uncertain/error result;
                          decodable frames        no fabricated score

  TC-08                   Score inside            "Uncertain" category
                          uncertainty band        shown

  TC-09                   Complete analysis       Temporary files are
                                                  removed according to
                                                  policy

  TC-10                   Backend unavailable     Frontend displays a
                                                  recoverable error
  -----------------------------------------------------------------------

## 4. Model metrics

Report: - Precision - Recall - F1-score - ROC-AUC - PR-AUC -
False-positive rate - False-negative rate - Confusion matrix

Accuracy may be included but should not be the only metric, particularly
when class balance differs.

## 5. Generalization tests

Where feasible, evaluate: - A dataset or manipulation technique not used
for training. - Lower resolution and different compression levels. -
Different lighting and image quality. - Images with no visible face or
multiple faces. - Videos with varied duration and frame rates.

Report the tested datasets and conditions explicitly. Do not imply broad
real-world reliability from a single benchmark.

## 6. Acceptance criteria

-   All critical input validation tests pass.
-   Valid image and video workflows complete without unhandled
    exceptions.
-   The UI communicates model score and uncertainty.
-   Evaluation is reproducible from recorded configuration.
-   Metrics and limitations are included in the final report.

## 7. Test evidence

Maintain a table of test ID, date, software/model version, result
(pass/fail), notes, and evidence location. Record failed tests and fixes
rather than omitting them.
