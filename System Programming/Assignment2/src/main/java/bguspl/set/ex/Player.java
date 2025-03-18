package bguspl.set.ex;

import java.util.Scanner;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

import bguspl.set.Env;

import java.util.ArrayList;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.List;

/**
 * This class manages the players' threads and data
 *
 * @inv id >= 0
 * @inv score >= 0
 */
public class Player implements Runnable {

    /**
     * The game environment object.
     */
    private final Env env;

    /**
     * Game entities.
     */
    private final Table table;

    /**
     * The id of the player (starting from 0).
     */
    public final int id;

    /**
     * The thread representing the current player.
     */
    private Thread playerThread;

    /**
     * The thread of the AI (computer) player (an additional thread used to generate key presses).
     */
    private Thread aiThread;

    /**
     * True iff the player is human (not a computer player).
     */
    private final boolean human;

    /**
     * True iff game should be terminated.
     */
    private volatile boolean terminate;
    private BlockingQueue<Integer> actions = new LinkedBlockingQueue<Integer>(3);

    private boolean inPenalty = false;
    private boolean inPoint = false;
    public boolean started = false;

    /**
     * The current score of the player.
     */
    private int score;
    private Dealer dealer;
    private long startTime = 0;
    private long elapsed = 0;

    private final long stop = 1000;
    private final long sleep = 950;
    private final long aiSleep = 100;

    /**
     * The class constructor.
     *
     * @param env    - the environment object.
     * @param dealer - the dealer object.
     * @param table  - the table object.
     * @param id     - the id of the player.
     * @param human  - true iff the player is a human player (i.e. input is provided manually, via the keyboard).
     */
    public Player(Env env, Dealer dealer, Table table, int id, boolean human) {
        this.env = env;
        this.table = table;
        this.id = id;
        this.human = human;
        this.dealer = dealer;
    }

    /**
     * The main player thread of each player starts here (main loop for the player thread).
     */
    @Override
    public void run() {
        playerThread = Thread.currentThread();
        started = true;
        env.logger.info("thread " + Thread.currentThread().getName() + " starting.");
        if (!human) createArtificialIntelligence();
        while (!terminate) {
            synchronized(actions){
                try{
                    actions.wait(); //waiting for key press
                }catch(InterruptedException e){}
                for (int slot: actions){
                    if (table.playerSlots[id][slot] == null)
                        table.placeToken(id, slot);
                    else 
                        table.removeToken(id, slot);
                    actions.remove();
                }
            }
            synchronized(this){
                if(table.playerSetSize(id) == env.config.featureSize){
                    dealer.enterQueue(this);
                    try{
                        this.wait(); //wait for dealer to respond
                    }catch(InterruptedException e){}
                }
            }
            startTime = System.currentTimeMillis();
            elapsed = System.currentTimeMillis() - startTime;
            if (inPenalty){   
                setCountdown(env.config.penaltyFreezeMillis);
                inPenalty = false;
            }
            else if (inPoint){
                setCountdown(env.config.pointFreezeMillis);
                inPoint = false;
            }
        }
        if (!human) try { aiThread.join(); } catch (InterruptedException ignored) {}
        env.logger.info("thread " + Thread.currentThread().getName() + " terminated.");
    }

    /**
     * Creates an additional thread for an AI (computer) player. The main loop of this thread repeatedly generates
     * key presses. If the queue of key presses is full, the thread waits until it is not full.
     */
    private void createArtificialIntelligence() {
        // note: this is a very, very smart AI (!)
        aiThread = new Thread(() -> {
            env.logger.info("thread " + Thread.currentThread().getName() + " starting.");
            while (!terminate) {
                try{
                    Thread.sleep(aiSleep); 
                }catch(InterruptedException e){}
                Random r = new Random();
                int rand = r.nextInt(env.config.tableSize);
                keyPressedAi(rand);
            }
            env.logger.info("thread " + Thread.currentThread().getName() + " terminated.");
        }, "computer-" + id);
        aiThread.start();
    }

    public void setCountdown(long time){
        while(elapsed <= time - stop){
            env.ui.setFreeze(id, time - elapsed);
            try{
                Thread.sleep(sleep); //sleeps for a second
            }catch(InterruptedException e){}
            elapsed = System.currentTimeMillis() - startTime;
        }
        env.ui.setFreeze(id, 0); 
    }

    /**
     * Called when the game should be terminated.
     */
    public void terminate() {
        synchronized(actions){
            actions.notify();
        }
        terminate = true;
    }

    /**
     * This method is called when a key is pressed.
     *
     * @param slot - the slot corresponding to the key pressed.
     */
    public void keyPressed(int slot) {
        if (human && !inPenalty && !inPoint){
            synchronized(actions){
                if (actions.size() < env.config.featureSize &&
                (table.playerSetSize(id) < env.config.featureSize || table.playerSlots[id][slot] != null)){ 
                        actions.add(slot);
                        actions.notifyAll();
                }
            }
        }
    }

    public void keyPressedAi(int slot){
        if (!inPenalty && !inPoint){
            synchronized(actions){
                if (actions.size() < env.config.featureSize &&
                (table.playerSetSize(id) < env.config.featureSize || table.playerSlots[id][slot] != null)){ 
                    actions.add(slot);
                    try{
                        actions.notifyAll();
                    }catch (IllegalMonitorStateException e){}
                }
            }
        }
    }

    /**
     * Award a point to a player and perform other related actions.
     *
     * @post - the player's score is increased by 1.
     * @post - the player's score is updated in the ui.
     */
    public void point() {
        inPoint = true;   
        actions = new LinkedBlockingQueue<Integer>(env.config.featureSize);
        env.ui.setScore(id, ++score);
    }

    /**
     * Penalize a player and perform other related actions.
     */
    public void penalty() {
        inPenalty = true;
    }

    public int score() {
        return score;
    }

    public void resetActions(){
        synchronized(actions){
            actions = new LinkedBlockingQueue<Integer>(env.config.featureSize);
        }
    }

    public Thread getThread(){
        return playerThread;
    }
}