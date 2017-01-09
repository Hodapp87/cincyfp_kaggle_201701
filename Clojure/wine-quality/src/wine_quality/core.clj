(ns wine-quality.core
  (:require [clojure.data.csv :as csv]
            [clojure.java.io :as io])
  (:import (weka.classifiers.trees REPTree)
           (weka.core.converters ConverterUtils$DataSource)
           (weka.classifiers Evaluation)
           (java.util Random)
           (weka.classifiers.evaluation EvaluationUtils)
           (weka.core Instance)))

;; This implemenation uses Weka this is one example programatically
;; using a Weka model to use another model, import the new class and
;; replace in the code

(def raw-data (with-open [in-file (io/reader "../../data/winequality-data.csv")]
                (doall
                 (csv/read-csv in-file))))

(def test-data (with-open [in-file (io/reader "../../data/winequality-solution-input.csv")]
                 (doall
                  (csv/read-csv in-file))))

(def arff-header
"@relation winequality

@attribute    fixed_acid           numeric
@attribute    volatile_acid        numeric
@attribute    citric_acid          numeric
@attribute    residual_sugar       numeric
@attribute    chlorides            numeric
@attribute    free_sulfur_dioxide  numeric
@attribute    total_sulfur_dioxide numeric
@attribute    density              numeric
@attribute    ph                   numeric
@attribute    sulphates            numeric
@attribute    alcohol              numeric
@attribute    quality              numeric

@data
")

(defn to-arff
  "Use this to write the data out to an Weka arff format."
  []
  (with-open [out-file (io/writer "wine-quality.arff")]
    (.write out-file arff-header)
    (csv/write-csv out-file (mapv drop-last (rest raw-data)))))

(defn train-classifier []
  (let [classifier (new REPTree)
        source (new ConverterUtils$DataSource "wine-quality.arff")
        data (.getDataSet source)
        _ (.setClassIndex data (dec (.numAttributes data)))
        _ (.buildClassifier classifier data)
        e (new Evaluation data)]
    (.crossValidateModel e
                         classifier
                         data
                         (.intValue (int 10))
                         (new Random 1)
                         (into-array []))
    (println (.toSummaryString e))
    {:evaluator e
     :data data
     :classifier classifier}))

(defn stats [ev]
  (println (.toSummaryString (:evaluator ev))))

(defn gen-instance [dataset values]
  (let [inst (new Instance 12)]
    (doall (map-indexed (fn [idx x] (.setValue inst idx (Double. x))) values))
    (doto inst
      (.setValue 11 (Instance/missingValue))
      (.setDataset dataset))))

(def test-data
  (mapv #(take 11 %) (rest raw-data)))

(defn predict [{:keys [evaluator classifier data]} vals]
  (.evaluateModelOnce evaluator classifier (gen-instance data vals)))

(defn to-kaggle-results [model]
  (with-open [out-file (io/writer "kaggle-results.csv")]
    (csv/write-csv out-file
                    (into [["id" "quality"]]
                          (mapv (fn [x] [(last x) (predict model (drop-last x))])
                                (rest test-data))))))

(comment

  ;; format the data to an arff file to explore the models in weka
  ;; http://www.cs.waikato.ac.nz/ml/weka/downloading.html
  (to-arff)
  ;; once you have model you like use can train it against the data
  ;; just like in weka
  (def model (train-classifier))
  (stats model)
  ;; next lets run the data and get the predicted results for the Kaggle

  (to-kaggle-results)
  )
