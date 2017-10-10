package com.example.arham.neuralnetwork;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Created by Arham on 06/10/2017.
 */

public class LSTMNetwork {

    private static int seed = 12345;
    private static int iterations = 1;
    private static double learningRate = 0.1;
    private static int exampleLength = 500;
    private static int lstmLayer1Size = 200;
    private static int lstmLayer2Size = 200;
    private static int denseLayerSize = 200;
    private static double dropoutRatio = 0.4;

    public static MultiLayerNetwork buildLstmNetworks (int nIn, int nOut) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .learningRate(learningRate)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.RMSPROP)
                .rmsDecay(0.95)
                .regularization(true)
                .l2(1e-4)
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(lstmLayer1Size).activation(Activation.TANH).gateActivationFunction(Activation.HARDSIGMOID).dropOut(dropoutRatio).build())
                .layer(1, new GravesLSTM.Builder().nIn(lstmLayer1Size).nOut(lstmLayer2Size).activation(Activation.TANH).gateActivationFunction(Activation.HARDSIGMOID).dropOut(dropoutRatio).build())
                .layer(2, new DenseLayer.Builder().nIn(lstmLayer2Size).nOut(denseLayerSize).activation(Activation.RELU).build())
                .layer(3, new RnnOutputLayer.Builder().nIn(denseLayerSize).nOut(nOut).activation(Activation.IDENTITY).lossFunction(LossFunctions.LossFunction.MSE).build())
                .backpropType(BackpropType.TruncatedBPTT) // trivial here, foreward and backward length are same as exampleLength
                .tBPTTForwardLength(exampleLength)
                .tBPTTBackwardLength(exampleLength)
                .pretrain(false)
                .backprop(true)
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(100));
        return net;
    }
}
