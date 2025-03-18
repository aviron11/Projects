package bguspl.set.ex;

import bguspl.set.Env;

import java.security.Key;
import java.util.List;
import java.util.Random;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * This class manages the dealer's threads and data
 */
public class Dealer implements Runnable {

    /**
     * The game environment object.
     */
    private final Env env;

    /**
     * Game entities.
     */
    private final Table table;
    private final Player[] players;

    /**
     * The list of card ids that are left in the dealer's deck.
     */
    private final List<Integer> deck;

    /**
     * True iff game should be terminated.
     */
    private volatile boolean terminate;
    private final long sleep = 100; //needs to update milliseconds during last 5 seconds
    private long startTime;
    private long elapsed = 0;

    private BlockingQueue<Player> queue = new LinkedBlockingQueue<Player>();

    /**
     * The time when the dealer needs to reshuffle the deck due to turn timeout.
     */
    private long reshuffleTime = Long.MAX_VALUE;

    public Dealer(Env env, Table table, Player[] players) {
        this.env = env;
        reshuffleTime = env.config.turnTimeoutMillis;
        this.table = table;
        this.players = players;
        deck = IntStream.range(0, env.config.deckSize).boxed().collect(Collectors.toList());
    }

    /**
     * The dealer thread starts here (main loop for the dealer thread).
     */
    @Override
    public void run() {
        env.logger.info("thread " + Thread.currentThread().getName() + " starting.");
        for (int i = 0; i < players.length; i++){ //creating player threads 
            Thread pl = new Thread(players[i]);
            pl.start();
            while (!players[i].started){ 
                try{
                    Thread.sleep(sleep); //making sure the threads are created by order
                }catch (InterruptedException e){}
            }
        }
        while (!shouldFinish()) {
            placeCardsOnTable();
            if(table.countCards() == 0)
                terminate = true;
            timerLoop();
            removeAllCardsFromTable();
        }
        announceWinners();
        terminate();
        env.logger.info("thread " + Thread.currentThread().getName() + " terminated.");
    }

    /**
     * The inner loop of the dealer thread that runs as long as the countdown did not time out.
     */
    private void timerLoop() {
        startTime = System.currentTimeMillis();
        elapsed = System.currentTimeMillis() - startTime;
        while (!terminate && elapsed < reshuffleTime) {
            updateTimerDisplay(false);
            sleepUntilWokenOrTimeout();
            removeCardsFromTable();
            placeCardsOnTable();
        }
        updateTimerDisplay(true);
    }

    /**
     * Called when the game should be terminated.
     */
    public void terminate() {
        synchronized(this){
            for (int i = players.length -1; i>=0; i--){ //terminates player threads in reversed order
                players[i].terminate();
                try{
                    players[i].getThread().join();
                }catch (InterruptedException e){}
            }
        }
        terminate = true;
    }

    /**
     * Check if the game should be terminated or the game end conditions are met.
     *
     * @return true iff the game should be finished.
     */
    private boolean shouldFinish() {
        return terminate || env.util.findSets(deck, 1).size() == 0;
    }

    /**
     * Checks cards should be removed from the table and removes them.
     */
    private void removeCardsFromTable() {
        if (!queue.isEmpty()){
            for (Player p : queue){
                    int[] pCards = table.playerSet(p.id);
                    if (pCards.length == env.config.featureSize){
                        if (env.util.testSet(pCards)){
                            p.point();
                            for (int card: pCards){
                                table.removeCard(table.cardToSlot[card]);
                            }
                            updateTimerDisplay(true);
                        }
                        else{
                            p.penalty();
                        }
                    }
                queue.remove(p);
                p.resetActions(); //clear player queue 
                synchronized(p){
                    p.notifyAll();
                }
            }
        }
    }

    /**
     * Check if any cards can be removed from the deck and placed on the table.
     */
    private void placeCardsOnTable() {
        for (int i = 0; i < table.slotToCard.length; i++)
        {
            if (table.slotToCard[i] == null)
            {
                if (!deck.isEmpty()){
                    Random rand = new Random();
                    int randNum = rand.nextInt(deck.size());
                    table.placeCard(deck.get(randNum), i);
                    deck.remove(randNum);
                }
            }
        }
    }

    /**
     * Sleep for a fixed amount of time or until the thread is awakened for some purpose.
     */
    private void sleepUntilWokenOrTimeout() {
        try{
            synchronized(queue){
                queue.wait(sleep);
            }
        }catch(InterruptedException e){}
    }

    /**
     * Reset and/or update the countdown and the countdown display.
     */
    private void updateTimerDisplay(boolean reset) {
        if (reset){
            startTime = System.currentTimeMillis();
            env.ui.setCountdown(env.config.turnTimeoutMillis, false);
            elapsed = System.currentTimeMillis() - startTime; 
        }
        long time =  env.config.turnTimeoutMillis - elapsed;
        env.ui.setCountdown(time, time <= env.config.turnTimeoutWarningMillis);       
        elapsed = System.currentTimeMillis() - startTime; 
    }

    /**
     * Returns all the cards from the table to the deck.
     */
    private void removeAllCardsFromTable() {
        for (int i = 0; i < table.slotToCard.length; i++){
            if (table.slotToCard[i] != null){
                deck.add(table.slotToCard[i]);
                table.removeCard(i);
            }
        }
    }

    /**
     * Check who is/are the winner/s and displays them.
     */
    private void announceWinners() {
        int maxScore = 0;
        for (Player p : players){
            if (p.score() > maxScore){
                maxScore = p.score();
            }
        }
        int count = 0;
        for (Player p : players){
            if (p.score() == maxScore){
                count++;
            }
        }
        int[] winners = new int[count]; 
        int index=0;
        for (Player p : players){
            if (p.score() == maxScore){
                winners[index] = p.id;
                index++;
            }
        }
        env.ui.announceWinner(winners);
    }

    public void enterQueue(Player p){
        queue.add(p);
        synchronized(queue){
            queue.notifyAll();
        }
    }
}