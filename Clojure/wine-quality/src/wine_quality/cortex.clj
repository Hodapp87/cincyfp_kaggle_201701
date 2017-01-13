(ns wine-quality.cortex
  (:require [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [cortex.dataset :as ds]
            [cortex.loss :as loss]
            [cortex.nn.execute :as execute]
            [cortex.nn.layers :as layers]
            [cortex.nn.network :as network]
            [cortex.nn.traverse :as traverse]
            [cortex.optimise :as opt]
            [think.compute.nn.compute-execute :as ce]
            [think.compute.nn.cpu-backend :as cpu-backend]
            [think.resource.core :as resource]
            [wine-quality.core :as wine-quality]))

;;; An attempt at using Cortex for the wine quality data
;;; https://github.com/thinktopic/cortex

;; Notes from Google Group

;; Just a couple of quick notes. This is the type of problem that is typically a better fit for a decision tree or regression approach, or an ensemble thereof like adaboost/gradient boosting or random forests. That said, a neural network with no hidden layers and a softmax output is essentially equivalent to logistic regression, or basic regression with a linear output, and could be a good starter example.

;; There would still be a few gotchas in approaching this dataset, but they could just as easily be teaching opportunities rather than problems. I'm thinking specifically that you'd probably want to take steps to balance the dataset (roughly equal number of samples per class per training batch) and normalize the input, e.g. to values from 0.0 to 1.0 or -1.0 to 1.0. It looks like mostly the input values are quantities rather than categorical, so you likely won't have to worry about encoding categorical variables. If I missed something in scanning though and there are categorical values, you'd probably want to encode those as dummies.

;; From there you could experiment with doing things like adding a hidden layer to see if feature combinations are helpful. You could also use an l2 penalty/constraint on the weights as a form of regularization. With a hold out validation and/or test dataset this is likely to make training and validation error more similar than they'd be w/o the l2 penalty.


(defn- train-and-get-results
  [context network input-bindings output-bindings
   batch-size dataset optimiser disable-infer? infer-batch-type
   n-epochs map-fn]
  (let [output-id (ffirst output-bindings)]
    (resource/with-resource-context
      (network/print-layer-summary (-> network
                                       network/build-network
                                       traverse/auto-bind-io
                                       traverse/network->training-traversal))
      (let [trained-network  (as-> (network/build-network network) net-or-seq
                               (execute/train context net-or-seq dataset input-bindings output-bindings
                                              :batch-size batch-size
                                              :optimiser optimiser
                                              :disable-infer? disable-infer?
                                              :infer-batch-type infer-batch-type)
                               (take n-epochs net-or-seq)
                               (map map-fn net-or-seq)
                               (last net-or-seq)
                               (execute/save-to-network context (get net-or-seq :network) {}))
            infer-results  (-> (execute/infer-columns context trained-network dataset input-bindings output-bindings
                                                      :batch-size batch-size)
                               (get output-id))]
        {:network trained-network :infer-results infer-results}))))


(defn create-context
  []
  (ce/create-context
   #(cpu-backend/create-cpu-backend :float)))


;;;;; Wine quality

(def WINE-DATA
  (->> (rest wine-quality/raw-data)
       (mapv #(take 11 %))
       (mapv #(mapv (fn [x] (Double/parseDouble x)) %))))

(def WINE-LABELS
  (->> (rest wine-quality/raw-data)
       (mapv #(nth % 11))
       (mapv #(Double. %))))

(def TEST-WINE-DATA
  (->> (rest wine-quality/test-data)
       (mapv #(mapv (fn [x] (Double. x)) %))))


(defn train-wine
  "This trains the network on the wine data and results the Mean squared loss"
  [context]
  (let [n-epochs 5000
        dataset (ds/create-in-memory-dataset {:data {:data WINE-DATA
                                                     :shape 11}
                                              :labels {:data WINE-LABELS
                                                       :shape 1}}
                                             (ds/create-index-sets (count WINE-DATA)
                                                                   :training-split 1.0
                                                                   :randomize? false))

        loss-fn (loss/mse-loss)
        input-bindings [(traverse/->input-binding :input :data)]
        output-bindings [(traverse/->output-binding :output
                                                    :stream :labels
                                                    :loss loss-fn)]
        results (train-and-get-results context [(layers/input 11 1 1 :id :input)
                                                (layers/linear 10 :id :hidden)
                                                (layers/linear 1 :id :output)]
                                       input-bindings output-bindings 100 dataset
                                       (opt/adadelta) true nil n-epochs identity)
        mse (loss/average-loss loss-fn (:infer-results results) WINE-LABELS)
        output-id (ffirst output-bindings)]
    (println "The MSE! for the wine training is " mse)
    (assoc results :mse mse)))


(defn test-wine
  "Gets the results of a trained network on the test data"
  [context trained-network]
  (let [observations (mapv drop-last TEST-WINE-DATA)
        dataset (ds/->InMemoryDataset {:data {:data observations
                                              :shape 11}}
                                      (vec (range (count observations))))
        infer-results  (execute/infer-columns context trained-network dataset [] [] :batch-size (count observations))]
    infer-results))

(defn to-kaggle-results
  "Writes the test results in the Kaggle format"
  [results]
  (with-open [out-file (io/writer "cortex-kaggle-results.csv")]
    (csv/write-csv out-file
                    (into [["id" "quality"]]
                          (mapv (fn [idx r]
                                  [(int  idx) (double r)]) (mapv last TEST-WINE-DATA) (mapv first results))))))

(comment

  (def trained-network (train-wine (create-context)))

  (def infer-results
    (test-wine (create-context) (:network trained-network)))

  (to-kaggle-results (:output infer-results))

  ;;; score 0.59625
)
