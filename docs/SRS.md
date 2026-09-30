# Software Requirements Specification (SRS)

## 1. Introduction

### 1.1 Purpose

This document defines the requirements for a semester-scale AI-Based
Deepfake Detection System. It provides a common reference for
implementation, testing, and project evaluation.

### 1.2 Scope

The system is a web-based prototype that analyzes uploaded images and
videos for visual indicators associated with facial manipulation. It
uses a pretrained image-classification model fine-tuned on labelled
examples. Video analysis is performed by sampling frames and aggregating
frame-level scores.

### 1.3 Intended users

-   Students and project evaluators demonstrating the system.
-   General users who want an initial automated screening of visual
    media.

### 1.4 Definitions

-   **Deepfake:** Media whose visual or audio content has been
    synthetically generated or manipulated.
-   **Inference:** Using a trained model to generate a prediction for
    new input.
-   **Frame sampling:** Selecting a subset of frames from a video for
    analysis.
-   **Model score:** Numerical output produced by a classifier; not
    necessarily a calibrated probability.
-   **Uncertain:** A result used when the score is near a configured
    decision threshold or analysis quality is insufficient.

## 2. Overall description

### 2.1 Product perspective

The system consists of a React frontend, a FastAPI backend, a
media-preprocessing component, and a PyTorch inference module.

### 2.2 Product functions

-   Accept image and video uploads.
-   Validate file type and size.
-   Preprocess images and sample video frames.
-   Run visual classification.
-   Aggregate frame-level predictions for video.
-   Present a result and basic supporting information.
-   Handle invalid or unsupported files with clear messages.

### 2.3 User characteristics

Users are expected to have basic familiarity with uploading files
through a web application. No machine-learning knowledge is required to
use the interface.

### 2.4 Operating environment

The prototype is intended to run on a local development machine or a
server capable of running Python and a supported PyTorch model. GPU
acceleration is optional; CPU inference may be slower.

### 2.5 Constraints and assumptions

-   Training data must be legally and ethically usable.
-   Dataset labels are treated as ground truth for experimental
    purposes, subject to dataset limitations.
-   The first version focuses on visual facial manipulation.
-   Large files and long videos may be restricted to keep processing
    feasible.
-   Model performance is dependent on the training data and cannot
    guarantee detection of unseen methods.

## 3. Functional requirements

  -----------------------------------------------------------------------
  ID                      Requirement             Priority
  ----------------------- ----------------------- -----------------------
  FR-01                   The system shall allow  Must
                          users to upload         
                          supported image files.  

  FR-02                   The system shall allow  Must
                          users to upload         
                          supported video files.  

  FR-03                   The system shall reject Must
                          unsupported, empty, or  
                          oversized files with a  
                          readable error.         

  FR-04                   The backend shall       Must
                          preprocess an image to  
                          the input dimensions    
                          and format expected by  
                          the model.              

  FR-05                   The backend shall       Must
                          sample a bounded number 
                          of frames from an       
                          uploaded video.         

  FR-06                   The system shall        Must
                          classify an image or    
                          sampled video frames    
                          using a trained model.  

  FR-07                   The system shall        Must
                          aggregate frame scores  
                          into one video-level    
                          score using a           
                          documented method.      

  FR-08                   The system shall        Must
                          display the result      
                          category and model      
                          score.                  

  FR-09                   The system shall show   Must
                          an uncertain result     
                          when configured         
                          thresholds or quality   
                          checks require it.      

  FR-10                   The system should       Should
                          display selected        
                          sampled frames or       
                          frame-level scores as   
                          supporting information. 

  FR-11                   The system may provide  Could
                          a downloadable analysis 
                          summary.                
  -----------------------------------------------------------------------

## 4. Non-functional requirements

  -----------------------------------------------------------------------
  ID                                  Requirement
  ----------------------------------- -----------------------------------
  NFR-01                              The interface should be
                                      understandable without technical
                                      training.

  NFR-02                              Upload and analysis errors should
                                      be communicated without exposing
                                      stack traces.

  NFR-03                              The backend shall enforce file
                                      type, size, and duration limits.

  NFR-04                              Uploaded media should be stored
                                      only as long as needed for analysis
                                      unless retention is explicitly
                                      configured.

  NFR-05                              The system should separate
                                      preprocessing, inference, API, and
                                      presentation code.

  NFR-06                              Results should identify that the
                                      model is fallible and not a
                                      forensic certification.

  NFR-07                              The prototype should support
                                      repeatable evaluation with fixed
                                      dataset splits and recorded model
                                      settings.

  NFR-08                              The system should avoid logging
                                      uploaded media contents or
                                      unnecessary personal data.
  -----------------------------------------------------------------------

## 5. External interface requirements

### 5.1 User interface

The interface shall provide an upload area, file details, analysis
progress, result category, model score, and a short interpretation
notice.

### 5.2 API

Proposed endpoint: - `POST /api/analyze` --- accepts one supported media
file and returns an analysis identifier, media type, result category,
score, and available evidence.

A health endpoint such as `GET /api/health` may be provided for local
testing.

### 5.3 Hardware interface

No specialized hardware is required. A compatible NVIDIA GPU may
accelerate model training and inference but is optional.

## 6. Data requirements

Training and evaluation data should include authentic and manipulated
examples, with variation in identities, sources, manipulation methods,
resolution, and compression. Splits should be separated by
source/identity where dataset metadata permits, reducing leakage between
training and testing.

## 7. Acceptance criteria

-   Supported image and video files can be submitted.
-   Invalid inputs are rejected safely.
-   Image inference returns a valid score and category.
-   Video processing samples frames within configured limits and returns
    an aggregate score.
-   Results are displayed in the frontend.
-   Evaluation metrics are reported on held-out data.
-   The interface clearly communicates uncertainty and limitations.

## 8. Risks and mitigations

-   **Dataset bias:** evaluate across available datasets and report
    limitations.
-   **Unseen manipulations:** describe results as screening signals, not
    definitive judgments.
-   **Compute constraints:** use a pretrained model, bounded frame
    sampling, and modest input resolution.
-   **Privacy:** minimize retention and avoid unnecessary collection.
-   **False positives/negatives:** report precision, recall, F1, ROC-AUC
    and error cases, not accuracy alone.
