package com.example.arham.neuralnetwork;

/**
 * Created by Arham on 05/10/2017.
 */

public class StockData {

    private String date; // date
    private String symbol; // stock name
    private double open; // open price
    private double close; // close price
    private double low; // low price
    private double high; // high price
    private double volume; // volume
    public StockData () {}
    public StockData (String date, String symbol, double open, double close, double low, double high, double volume) {
        this.date = date;
        this.symbol = symbol;
        this.open = open;
        this.close = close;
        this.low = low;
        this.high = high;
        this.volume = volume;
    }

    public String getSymbol() {
        return symbol;
    }

    public void setSymbol(String s) {
        symbol = s;
    }

    public String getDate() {
        return date;
    }

    public void setDate(String d) {
        date = d;
    }

    public double getOpen() {
        return open;
    }

    public void setOpen(double o) {
        open  = o;
    }

    public double getClose() {
        return close;
    }

    public void setClose(double c) {
        close = c;
    }

    public double getLow() {
        return low;
    }

    public void setLow(double l) {
        low = l;
    }

    public double getHigh() {
        return high;
    }

    public void setHigh(double h) {
        high = h;
    }


    public double getVolume() {
        return volume;
    }

    public void setVolume(double v) {
        volume = v;
    }
}
