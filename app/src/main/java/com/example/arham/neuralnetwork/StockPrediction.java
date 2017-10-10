package com.example.arham.neuralnetwork;

import android.app.DownloadManager;
import android.content.Context;
import android.content.Intent;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.TextView;

import com.jjoe64.graphview.GraphView;
import com.jjoe64.graphview.LegendRenderer;
import com.jjoe64.graphview.series.DataPoint;
import com.jjoe64.graphview.series.LineGraphSeries;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.logging.Logger;

public class StockPrediction extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_stock_prediction);
        Log.d("testing", "Starting");
        //downloadData();
        TextView btn = (TextView) findViewById(R.id.btn);
        btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                try {
                    makePrediction();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        });

    }

    //private static final Logger log = (Logger) LoggerFactory.getLogger(StockPrediction.class);

    private static int batchSize = 500;
    private static int exampleLength = 500;
    private static double splitRatio = 0.75;

    public void makePrediction() throws IOException {
        String filename = "/storage/emulated/0/Download/acccmm.csv";    //download karday online
        String symbol = "ACCM"; // stock name                                               //https://www.dropbox.com/sh/lllir1mktozbkin/AACxH3xHrP91SIqcLAdeDzlja?dl=0&preview=ACCM.xls
        // create dataset iterator
        //log.info("create stock dataSet iterator...");
        Log.d("testing", "Going to iterator");
        Iterator iterator = new Iterator(filename, symbol, batchSize, exampleLength, splitRatio);
        //log.info("load test dataset...");
        Log.d("testing", "Loading test dataset");
        List<Pair<INDArray, Double>> test = iterator.getTestDataSet();
        // build lstm network
        //log.info("build lstm networks...");
        Log.d("testing", "Building LSTM Network");
        MultiLayerNetwork net = LSTMNetwork.buildLstmNetworks(iterator.inputColumns(), iterator.totalOutcomes());
        // training
        //log.info("training...");
        Log.d("testing", "Training");
        for (int i = 0; i < 1; i++) {
            Log.d("testing", "For loop entered");
            DataSet dataSet;
            Log.d("testing", "Dataset declared");
            while (iterator.hasNext()) {
                Log.d("testing", "while loop");
                dataSet = iterator.next();
                Log.d("testing", "next");
                INDArray x = dataSet.getFeatures();
                Log.d("testing", Double.toString(x.getDouble(0,0)));
                Log.d("testing", "dataset = " + dataSet.get(0));
                net.fit(dataSet);

                Log.d("testing", "fit");
            }
            Log.d("testing", "while loop done");
            iterator.reset(); // reset iterator
            Log.d("testing", "reset");
            net.rnnClearPreviousState(); // clear previous state
            Log.d("testing", "clear prev state");
        }
        // save model
        //log.info("saving model...");
        Log.d("testing", "Saving");
        /**File locationToSave = new File("src/main/res/StockPriceLSTM.zip");
        boolean saveUpdater = true; //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        ModelSerializer.writeModel(net, locationToSave, saveUpdater);**/
        // load model
        //log.info("load model...");
        //MultiLayerNetwork restoredNet = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
        // Testing
        //log.info("testing...");
        Log.d("testing", "Testing");
        double max = iterator.getMaxNum()[1];
        double min = iterator.getMinNum()[1];
        double[] predicts = new double[test.size()];
        double[] actuals = new double[test.size()];
        for (int i = 0; i < test.size(); i++) {
            predicts[i] = net.rnnTimeStep(test.get(i).getFirst()).getDouble(exampleLength - 1) * (max - min) + min;
            actuals[i] = test.get(i).getSecond();
        }
        // print out
        /**System.out.println(String.format("%20s", "Predict") + String.format("%20s", "Actual"));
        for (int i = 0; i < predicts.length; i++) {
            System.out.println(String.format("%20s", String.valueOf(predicts[i])) + String.format("%20s", String.valueOf(actuals[i])));
        }**/
        // plot
        //PlotUtil.plot(predicts, actuals);
        Log.d("testing", "Plotting graph");
        GraphView graph = (GraphView) findViewById(R.id.graph);

        DataPoint[] points = new DataPoint[500];
        Log.d("testing", "Adding points");
        for (int j = 0; j < points.length; j++) {
            Log.d("testing", "abc");
            Log.d("testing", String.valueOf(predicts[j]));
            points[j] = new DataPoint(j, predicts[j]);
            Log.d("testing", "Point" + String.valueOf(j));
        }
        LineGraphSeries<DataPoint> series = new LineGraphSeries<>(points);

        DataPoint[] points2 = new DataPoint[500];
        for (int k = 0; k < points2.length; k++) {
            points2[k] = new DataPoint(k, actuals[k]);
        }
        LineGraphSeries<DataPoint> series2 = new LineGraphSeries<>(points2);
        Log.d("testing", "Points done");
        series.setTitle("prediction");
        series2.setTitle("actual");
        graph.getLegendRenderer().setVisible(true);
        graph.getLegendRenderer().setAlign(LegendRenderer.LegendAlign.TOP);

        graph.getViewport().setScalable(true);
        graph.getViewport().setScalableY(true);

        series.setColor(Color.BLUE);
        graph.addSeries(series);

        series2.setColor(Color.RED);
        graph.addSeries(series2);

    }

    public void downloadData() {
        Log.d("testing", "Starting to download");
        String url = "https://www.dropbox.com/s/z30hyg9ffwved1s/acccm.csv?dl=0";
        Log.d("testing", "Url hogayi");
        DownloadManager.Request request = new DownloadManager.Request(Uri.parse(url));
        request.setDescription("Market data");
        request.setTitle("accm");
        Log.d("testing", "set hogaya");
        // in order for this if to run, you must use the android 3.2 to compile your app
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.HONEYCOMB) {
            request.allowScanningByMediaScanner();
            request.setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED);
        }
        Log.d("testing", "yahan");
        request.setDestinationInExternalPublicDir(Environment.DIRECTORY_DOWNLOADS, "accm.csv");
        Log.d("testing", "directory hogayi");
        // get download service and enqueue file
        DownloadManager manager = (DownloadManager) getSystemService(Context.DOWNLOAD_SERVICE);
        Log.d("testing", "service");
        manager.enqueue(request);
        Log.d("testing", "Downloaded");
    }
}
