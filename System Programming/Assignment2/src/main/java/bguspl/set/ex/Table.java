package bguspl.set.ex;

import bguspl.set.Env;
import bguspl.set.UserInterface;
import bguspl.set.UserInterfaceDecorator;
import bguspl.set.UtilImpl;

import java.io.File;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.zip.Inflater;

/**
 * This class contains the data that is visible to the player.
 *
 * @inv slotToCard[x] == y iff cardToSlot[y] == x
 */
public class Table {

    /**
     * The game environment object.
     */
    private final Env env;

    /**
     * Mapping between a slot and the card placed in it (null if none).
     */
    protected final Integer[] slotToCard; // card per slot (if any)

    /**
     * Mapping between a card and the slot it is in (null if none).
     */
    protected final Integer[] cardToSlot; // slot per card (if any)
    protected Integer[][] playerSlots;
    private Object[] locks;

    /**
     * Constructor for testing.
     *
     * @param env        - the game environment objects.
     * @param slotToCard - mapping between a slot and the card placed in it (null if none).
     * @param cardToSlot - mapping between a card and the slot it is in (null if none).
     */
    public Table(Env env, Integer[] slotToCard, Integer[] cardToSlot) {
        this.env = env;
        this.slotToCard = slotToCard;
        this.cardToSlot = cardToSlot;
        playerSlots = new Integer[env.config.players][env.config.tableSize];
        locks = new Object[env.config.tableSize]; //so we can synchronize each slot, object[i] locks the i-th slot
        for (int i = 0; i < locks.length; i++){
            locks[i] = new Object(); 
        }
    }

    /**
     * Constructor for actual usage.
     *
     * @param env - the game environment objects.
     */
    public Table(Env env) {
        this(env, new Integer[env.config.tableSize], new Integer[env.config.deckSize]);
        playerSlots = new Integer[env.config.players][env.config.tableSize];
    }

    /**
     * This method prints all possible legal sets of cards that are currently on the table.
     */
    public void hints() {
        List<Integer> deck = Arrays.stream(slotToCard).filter(Objects::nonNull).collect(Collectors.toList());
        env.util.findSets(deck, Integer.MAX_VALUE).forEach(set -> {
            StringBuilder sb = new StringBuilder().append("Hint: Set found: ");
            List<Integer> slots = Arrays.stream(set).mapToObj(card -> cardToSlot[card]).sorted().collect(Collectors.toList());
            int[][] features = env.util.cardsToFeatures(set);
            System.out.println(sb.append("slots: ").append(slots).append(" features: ").append(Arrays.deepToString(features)));
        });
    }

    /**
     * Count the number of cards currently on the table.
     *
     * @return - the number of cards on the table.
     */
    public int countCards() {
        int cards = 0;
        for (Integer card : slotToCard)
            if (card != null)
                ++cards;
        return cards;
    }

    /**
     * Places a card on the table in a grid slot.
     * @param card - the card id to place in the slot.
     * @param slot - the slot in which the card should be placed.
     *
     * @post - the card placed is on the table, in the assigned slot.
     */
    public void placeCard(int card, int slot) {
        try {
            Thread.sleep(env.config.tableDelayMillis);
        } catch (InterruptedException ignored) {}

        cardToSlot[card] = slot;
        slotToCard[slot] = card;
        env.ui.placeCard(card, slot);
    }

    /**
     * Removes a card from a grid slot on the table.
     * @param slot - the slot from which to remove the card.
     */
    public void removeCard(int slot) {
        try {
            Thread.sleep(env.config.tableDelayMillis);
        } catch (InterruptedException ignored) {}
        synchronized(locks[slot]){ //we don't want the players placing tokens when the card is removed
            if (slotToCard[slot] != null)
                removeAllTokensInSlot(slot);
            cardToSlot[slotToCard[slot]] = null;
            slotToCard[slot]=null;
            env.ui.removeCard(slot);  
        }
    }

    /**
     * Places a player token on a grid slot.
     * @param player - the player the token belongs to.
     * @param slot   - the slot on which to place the token.
     */
    public void placeToken(int player, int slot) {
        synchronized(locks[slot]){
            if (slotToCard[slot] != null){
                env.ui.placeToken(player, slot);
                playerSlots[player][slot] = 1;
            }
        }
    }

    /**
     * Removes a token of a player from a grid slot.
     * @param player - the player the token belongs to.
     * @param slot   - the slot from which to remove the token.
     * @return       - true iff a token was successfully removed.
     */
    public boolean removeToken(int player, int slot) {
        if (playerSlots[player][slot] == null)
            return false;
        env.ui.removeToken(player, slot);
        playerSlots[player][slot] = null;
        return true;
    }

    public void removeAllTokens(){
        for(int i = 0; i < env.config.tableSize; i++){
            removeAllTokensInSlot(i);
        }   
    }

    public void removeAllTokensInSlot(int slot){
        for (int i = 0; i < env.config.players; i++){
            removeToken(i, slot);
            playerSlots[i][slot] = null;
        }
    }

    public void removeAllPlayerTokens(int id){
        for (int i = 0; i < env.config.tableSize; i++){
            playerSlots[id][i] = null;
            removeToken(id, i);
        }
    }

    //returns the number of tokens the player currently has on the table
    public int playerSetSize(int id){
        int counter = 0;
        for (int i = 0; i < env.config.tableSize; i++){
            if (playerSlots[id][i] != null)
                counter++;
        }
        return counter;
    }

    //returns the set of cards the player chose
    public int[] playerSet(int id){
        int size = playerSetSize(id);
        int[] output = new int[size];
        int index = 0;
        for (int i = 0; i < env.config.tableSize; i++){
            if (playerSlots[id][i] != null){
                output[index]=slotToCard[i];
                index++;
            }
        }
        return output;
    }
}