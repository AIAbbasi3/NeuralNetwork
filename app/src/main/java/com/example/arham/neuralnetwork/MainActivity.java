package com.example.arham.neuralnetwork;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import org.datavec.api.berkeley.Pair;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

import static android.R.attr.button;
import static android.R.id.button1;

public class MainActivity extends AppCompatActivity {

    public void init() {
        TextView button1 = (TextView) findViewById(R.id.textView);
        Log.d("testing", "Start");
        button1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Log.d("testing", "Clicked");
                Intent viewMkt = new Intent(MainActivity.this,StockPrediction.class);
                Log.d("testing", "Next");
                startActivity(viewMkt);
                Log.d("testing", "Started");
            }
        });

    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        init();

        /**try {
            createNetwork();
        } catch (IOException e) {
            e.printStackTrace();
        }**/
    }

    /**public void loadData() throws IOException, InterruptedException {
        SequenceRecordReader reader = new CSVSequenceRecordReader(1, ",");
        reader.initialize(new NumberedFileInputSplit("C:/Users/Arham/Desktop/FYP 2/data/PSX Historical Data/accmcsv"));
        DataSetIterator iterClassification = new SequenceRecordReaderDataSetIterator(reader, miniBatchSize, numPossibleLabels, labelIndex, false);
    }**/


}
