# BTB.py

# ┌───────────┐
# │ IMPORTS : │
# └───────────┘

from typing import List, Optional, Tuple
import math

# ┌─────────────┐
# │ VARIABLES : │
# └─────────────┘

# Maximum number of bits for a tag :
TAG_BITS_LIMIT = 32
# Maximum number of bits for a set index :
INDEX_BITS_LIMIT = 20
# Maximum number of bits for a confidence saturating counter :
CONFIDENCE_COUNTER_BITS_LIMIT = 8
# Maximum number of total entries :
NUM_ENTRIES_LIMIT = 1048576 # = 2^20
# Maximum associativity :
ASSOCIATIVITY_LIMIT = 256

# ┌─────────────┐
# │ FUNCTIONS : │
# └─────────────┘

def xor_fold(address: int, index_bits: int) -> int:
    """
    XOR fold an address to produce a set index for BTB
    branch identification.
    This is a simple XOR-folding, which works by splitting
    the address into chunks of size index_bits and then XOR-ing
    them together two by two into a single chunk.
    Note : XOR is a non destructive operation.
    
    Args:
        - address [int] : branch instruction address to XOR-fold
        - index_bits [int] : how many bits are needed for the index
    
    Returns:
        - index [int] : set index (0 to 2^index_bits - 1)
    """

    # Create a mask for extracting the index_bits lower bits of address :
    mask = (1 << index_bits) - 1 # the mask is 0 and then index_bits ones
    
    # Initialize index :
    index = 0
    
    # Keep XORing chunks until address is exhausted :
    while address != 0:
        chunk = address & mask # extract the lowest index_bits bits from address into a chunk
        index ^= chunk # XOR this chunk into the index
        address >>= index_bits # shift address right by index_bits bits to get next chunk

    return index

# ┌───────────┐
# │ CLASSES : │
# └───────────┘

class SaturatingCounter:
    """
    N-bit saturating counter.
    A counter that increments and decrements but saturates at
    minimum/maximum values instead of wrapping around.

    Attr :
        - n_bits [int] : number of bits
        - initial_value [int] : initialization value of the counter
    """
    
    def __init__(self, n_bits: int = 2, initial_value: int = 0):
        self.n_bits = n_bits
        self.max_value = (1 << n_bits) - 1 # 2^n_bits - 1
        self.min_value = 0

        # Check that the number of bits is correct :
        if n_bits <= 0:
            raise ValueError(f"Number of bits n_bits in saturating counter initialization must be a non-zero positive integer, got : {n_bits}\n")
        # Check that the initial value is correct :
        if initial_value < 0 or initial_value > self.max_value:
            raise ValueError(f"Initial value of saturating counter is invalid, got : {initial_value}, must be in : [{self.min_value} ; {self.max_value}]\n")

        # Initialize the counter with its initialization value :
        self.value = max(self.min_value, min(initial_value, self.max_value))
    
    def increment(self):
        """
        Increment counter (saturating at maximum)
        """

        if self.value < self.max_value:
            self.value += 1
    
    def decrement(self):
        """
        Decrement counter (saturating at minimum)
        """

        if self.value > self.min_value:
            self.value -= 1
    
    def reset(self, value: int = 0):
        """
        Reset counter to specified value

        Args :
            - value [int] : the value at which to reset the counter
        """

        if value < self.min_value or value > self.max_value:
            raise ValueError(f"Saturating counter was reset to invalid value, got : {value}, must be in : [{self.min_value}, {self.max_value}]\n")

        self.value = max(self.min_value, min(value, self.max_value))
    
    def is_higher(self, threshold: Optional[int] = None) -> bool:
        """
        Check if counter is at or above threshold
        
        Args :
            - threshold [Optional[int]] : value to compare against (default : midpoint)
        
        Returns :
            - [bool] : True if counter >= threshold
        """

        if threshold is None:
            threshold = self.max_value // 2 # use midpoint if no threshold is specified

        return self.value >= threshold
    
    def is_saturated_high(self) -> bool:
        """
        Check if counter is at maximum

        Returns :
            - [bool] : True if counter is at maximum
        """

        return self.value == self.max_value
    
    def is_saturated_low(self) -> bool:
        """
        Check if counter is at minimum

        Returns :
            - [bool] : True if counter is at minimum
        """
        return self.value == self.min_value
    
    def __repr__(self):
        return f"SaturatingCounter({self.n_bits}-bit, value={self.value}/{self.max_value})"
    
    def __int__(self):
        """
        Allow direct integer conversion

        Returns :
            - self.value [int] : the counter's value as an integer
        """
        return self.value
    
    def __eq__(self, other):
        """
        Allow comparison with integers or other counters
        based on the value of the counter.

        Returns :
            - [bool] : True if the two values are equal
        """

        # Counter/int comparison :
        if isinstance(other, int):
            return self.value == other
        # Counter/counter value comparison :
        elif isinstance(other, SaturatingCounter):
            return self.value == other.value

        return False
    
    def __lt__(self, other):
        """
        Allow less-than comparison with integers or other
        counters based on the value of the counter

        Returns :
            - [bool] : True if this counter's value is less than the other
        """

        # Counter/int comparison :
        if isinstance(other, int):
            return self.value < other
        # Counter/counter comparison :
        elif isinstance(other, SaturatingCounter):
            return self.value < other.value

        return NotImplemented
    
    def __le__(self, other):
        """
        Allow less-than-equal comparison with integers or other
        counters based on the value of the counter

        Returns :
            - [bool] : True if this counter's value is less than 
                       or equal to the other
        """

        # Counter/int comparison :
        if isinstance(other, int):
            return self.value <= other
        # Counter/counter comparison :
        elif isinstance(other, SaturatingCounter):
            return self.value <= other.value

        return NotImplemented

class BTBEntry:
    """
    Stores one entry of the BTB, a mapping between an instruction and
    its predicted target address.

    Attr :
        - valid [bool] : whether or not the entry is occupied
        - tag [int] : hash of address of instruction to identify branch
        - target [int] : address of target prediction
        - confidence [int] : n-bit (default 2) saturation counter of prediction confidence
    """

    def __init__(self, valid: bool = False, tag: int = 0, target: int = 0, confidence_counter_bits: int = 2, confidence: int = 0):
        self.valid = valid
        self.tag = tag
        self.target = target
        self.confidence_counter_bits = confidence_counter_bits
        self.confidence = SaturatingCounter(n_bits = confidence_counter_bits, initial_value = confidence)
    
    def __repr__(self):
        return f"BTBEntry(valid={self.valid}, tag={self.tag}, target={self.target}, confidence={self.confidence.value})"
    
    def __eq__(self, other):
        # If the two compared classes are different, fail with NotImplemented :
        if other.__class__ is not self.__class__:
            return NotImplemented
        # If two BTBEntry instances are being compared, compare that their are equal :
        return (self.valid, self.tag, self.target, self.confidence.value) == (other.valid, other.tag, other.target, other.confidence.value)

class BTBSet:
    """
    A single set containing multiple ways.
    Ways keep track of pseudo-LRU state for eviction when
    the set is full.

    Attr :
        - associativity [int] : the number of ways in a set
    """
    
    def __init__(self, associativity: int = 4, confidence_counter_bits: int = 2, default_confidence: int = 0):
        self.associativity = associativity # number of ways
        self.confidence_counter_bits = confidence_counter_bits
        self.default_confidence = default_confidence
        self.ways = [BTBEntry(confidence_counter_bits = self.confidence_counter_bits, confidence = self.default_confidence) for _ in range(associativity)]
        self.lru_bits: int = 0 # int that acts as pseudo-LRU tree (for n ways, needs (n-1) bits)
    
    def find_entry(self, tag: int) -> Optional[int]:
        """
        Search for an entry in the set that matches the tag.
        
        Args :
            - tag [int] : the tag to search for
            
        Returns :
            - way_num [Optional[int]] : way number if found, None if not found
        """

        for way_num in range(self.associativity):
            entry = self.ways[way_num]
             # If tags match, return way number :
            if entry.valid and entry.tag == tag:
                return way_num

        # If no match was found, return None :
        return None
    
    def find_invalid_way(self) -> Optional[int]:
        """
        Find an empty (invalid) way in the set.
        
        Returns :
            - way_num [Optional[int]] : way number if found, None if not found
        """

        for way_num in range(self.associativity):
            # If a way is not valid, return its number :
            if not self.ways[way_num].valid:
                return way_num

        # If all ways were valid, return None :
        return None

    def find_lru_way(self) -> int:
        """
        Pseudo-LRU (Least Recently Used) way finder using a binary
        tree. While it does not find the exactly LRU way, it is way
        faster than keeping a total ordering, and is decently correct.

        Finds the least recently used way index using pseudo-LRU bits :
        The lru_bits encode a binary tree that we can navigate ;
        The most recently used subtree is right if the bit is 1, left
        if it is 0.

        As we go up from the leaves, we pair the subtrees together ;
        to find the LRU way, we explore it down, always going towards
        the least recently used subtree, until we find the way.

        Returns :
            - index of the LRU way (4 <=> way 4 was LRU)
        """

        # Start at the beginning, considering all ways :
        start = 0
        end = self.associativity - 1
        bit_position = 0
        
        # Keep narrowing down until a single way is left :
        while start < end: # when start == end, the LRU way is identified
            # Compute midpoint to split into left and right subtrees :
            mid = (start + end) // 2
            # Check the value of the bit at the current position :
            bit_value = (self.lru_bits >> bit_position) & 1
            if bit_value == 1:
                # If bit is 1, right subtree is more recent, LRU must be in left subtree :
                end = mid
            else:
                # If bit is 0, left subtree is more recent, LRU must be in right subtree :
                start = mid + 1
            # Move to the next bit for the next level :
            bit_position += 1

        # Return the LRU way :
        return start

    def update_lru_bits(self, way: int, operation: str) -> None:
        """
        Update pseudo-LRU bits after a way is accessed or removed
        
        Args :
            - way [int] : which way index to update (0 to n-1)
            - operation [str] :
                - "mru" : make the way the most recently used one 
                - "rmv" : remove the way from the tree
        """

        # Start at the beginning, considering all ways :
        start = 0
        end = self.associativity - 1
        bit_position = 0
        
        # Navigate down the tree to the target way :
        while start < end:
            # Compute midpoint to split into left and right subtrees :
            mid = (start + end) // 2
            
            if operation == "mru":
                # Make target be the most recently used <=> set bits to point towards the target :
                if way <= mid:
                    # Target is in left subtree -> make bit 0 to point towards left :
                    self.lru_bits &= ~(1 << bit_position) # clear bit
                    end = mid
                else:
                    # Target is in right subtree -> make bit 1 to point towards right :
                    self.lru_bits |= (1 << bit_position) # set bit
                    start = mid + 1
            
            elif operation == "rmv":
                # Remove way -> make it LRU <=> set bits to point away from the target :
                if way <= mid:
                    # Target is in left subtree -> make bit 1 to point towards right :
                    self.lru_bits |= (1 << bit_position) # set bit
                    end = mid
                else:
                    # Target is in right subtree -> make bit 0 to point towards left :
                    self.lru_bits &= ~(1 << bit_position) # clear bit
                    start = mid + 1
            
            # If the operation passed isn't recognized :
            else:
                raise ValueError(f"Invalid operation when updating LRU bits : {operation}.\nMust be 'mru' or 'rmv'\n")
            
            # Move to next bit for next level :
            bit_position += 1

class BTB:
    """
    Branch Target Buffer with set-associative organization, pseudo-LRU
    per-set eviction and XOR-folding hash indexing.
    Ways are indexed by a tag, which is extracted from the XOR-folding
    of the address.

    Attr :
        - num_entries [int] : total number of ways
        - associativity [int] : number of ways per set (power of 2)
        - tag_bits [int] : number of bits used for tag
        - confidence_counter_bits [int] : number of bits to use for confidence
                                          counters of entries
        - default_confidence [int] : default confidence value for when a new
                                     entry is allocated (and not when it is
                                     initially created)
    """
    
    def __init__(self, num_entries: int = 4096, associativity: int = 4, tag_bits: int = 16, confidence_counter_bits: int = 2, default_confidence: int = 1):
        
        # Validate arguments :

        # Ensure that number of entries is positive :
        if num_entries <= 0:
            raise ValueError(f"BTB initialization error : num_entries must be positive, got {num_entries}")
        # Ensure that number of entries is below limit :
        if num_entries > NUM_ENTRIES_LIMIT:
            raise ValueError(f"BTB initialization error : num_entries must be <= {NUM_ENTRIES_LIMIT}, got {num_entries}")
        # Ensure that associativity is positive :
        if associativity <= 0:
            raise ValueError(f"BTB initialization error : associativity must be positive, got {associativity}")
        # Ensure that associativity is below limit :
        if associativity > ASSOCIATIVITY_LIMIT:
            raise ValueError(f"BTB initialization error : associativity must be <= {ASSOCIATIVITY_LIMIT}, got {associativity}")
        # Ensure associativity is a power of 2 (for pseudo-LRU to work correctly)
        if associativity & (associativity - 1) != 0:
            raise ValueError(f"BTB initialization error : associativity must be a power of 2, got {associativity}")
        # Ensure that the number of entries is a multiple of the associativity : 
        if num_entries % associativity != 0:
            raise ValueError("BTB initialization error : num_entries must be divisible by associativity")
        # Ensure that number of bits for tag is positive :
        if tag_bits <= 0:
            raise ValueError(f"BTB initialization error : tag_bits must be positive, got {tag_bits}")
        # Ensure that number of bits for tag is below limit :
        if tag_bits > TAG_BITS_LIMIT:
            raise ValueError(f"BTB initialization error : tag_bits must be <= {TAG_BITS_LIMIT}, got {tag_bits}")
        # Ensure that number of bits for confidence counter is at least 1 :
        if confidence_counter_bits < 1:
            raise ValueError(f"BTB initialization error : confidence_counter_bits must be at least 1, got {confidence_counter_bits}")
        # Ensure that number of bits for confidence counter is below limit :
        if confidence_counter_bits > CONFIDENCE_COUNTER_BITS_LIMIT:
            raise ValueError(f"BTB initialization error : confidence_counter_bits should be <= {CONFIDENCE_COUNTER_BITS_LIMIT}, got {confidence_counter_bits}")
        # Validate default_confidence is within valid range for the counter :
        max_confidence = (1 << confidence_counter_bits) - 1
        if default_confidence < 0 or default_confidence > max_confidence:
            raise ValueError(f"BTB initialization error : default_confidence must be in [0, {max_confidence}], got {default_confidence}")

        self.num_entries = num_entries
        self.associativity = associativity
        self.tag_bits = tag_bits
        self.confidence_counter_bits = confidence_counter_bits
        self.default_confidence = default_confidence

        # Compute the number of sets :
        self.num_sets = num_entries // associativity

        # Validate that number of sets is below limit :
        if self.num_sets > (1 << INDEX_BITS_LIMIT):
            raise ValueError(f"BTB initialization error : computed num_sets ({self.num_sets}) exceeds maximum allowed ({1 << INDEX_BITS_LIMIT})")
        
        # Compute how many bits are needed for set indexing (index_bits = ⌈log₂(num_sets)⌉):
        self.index_bits = int(math.ceil(math.log2(self.num_sets)))
        
        # Validate that number of bits for set indexes is below limit :
        if self.index_bits > INDEX_BITS_LIMIT:
            raise ValueError(f"BTB initialization error : computed index_bits ({self.index_bits}) exceeds maximum allowed ({INDEX_BITS_LIMIT})")

        # Create all sets :
        self.sets: List[BTBSet] = [BTBSet(associativity = self.associativity, confidence_counter_bits = self.confidence_counter_bits, default_confidence = 0) for _ in range(self.num_sets)]
        
        # Statistics for analysis :
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.updates = 0
    
    def _decompose_address(self, address: int) -> Tuple[int, int]:
        """
        Decompose branch address into set index and tag using consistent XOR-folding.
        Both index and tag are derived from the full address using XOR-folding, for
        good distribution and anti-aliasing (conflict minimization).

        Args :
            - address [int] : branch instruction address
            
        Returns :
            - (set_index, tag) [Tuple[int, int]] : the set index and tag
        """

        # Validate input
        if address < 0:
            raise ValueError(f"BTB address error : address must be non-negative, got {address}")
        
        # Remove byte offset (instructions are 4-byte aligned)
        address = address >> 2
        
        # Use XOR-folding to compute set index down to index_bits width
        set_index = xor_fold(address, self.index_bits) % self.num_sets  # ← Add modulo
        
        # Use XOR-folding to compute tag down to tag_bits width :
        tag = xor_fold(address, self.tag_bits)
        
        return set_index, tag

    def predict(self, branch_address: int) -> Optional[int]:
        """
        Look up a branch in the BTB and return predicted target.
        
        Args :
            - branch_address [int] : address of the branch instruction
            
        Returns :
            - predicted target address if found, None if miss
        """

        # Decompose address into set index and tag :
        set_index, tag = self._decompose_address(branch_address)
        
        # Get the relevant set :
        btb_set = self.sets[set_index]
        
        # Search for a matching entry in the set :
        way_num = btb_set.find_entry(tag)
        
        # BTB Hit :
        if way_num is not None:
            self.hits += 1 # update statistics
            # Update LRU state (mark way found as most recently used) :
            btb_set.update_lru_bits(way_num, "mru")
            # Return predicted target :
            return btb_set.ways[way_num].target
        
        # BTB miss :
        self.misses += 1 # update statistics
        # Return None as no match was found :
        return None
    
    def update(self, branch_address: int, actual_target: int):
        """
        Update confidence counters and/or update target of BTB entry
        based on what has been detected.

        If the entry doesn't exist for this address, create it (and
        evict if necessary).
        If it already exists for this address :
        If actual_target is the same as the one currently stored, then
        this counts as a BTB hit, and reinforces confidence.
        If it is different, then :
        - if BTB entry has reached 0 confidence, update the target to correct
        - if BTB entry has non-0 confidence, lower it
        
        Args :
            - branch_address [int] : address of the branch instruction
            - actual_target [int] : the detected target that should be predicted
        """

        self.updates += 1 # update statistics

        # Decompose address into set index and tag :
        set_index, tag = self._decompose_address(branch_address)
        
        # Get the relevant set :
        btb_set = self.sets[set_index]
        
        # Check if entry already exists :
        existing_way_num = btb_set.find_entry(tag)
        
        # If the entry exists, update it :
        if existing_way_num is not None:

            # Make it most recently used :
            btb_set.update_lru_bits(existing_way_num, "mru")

            # If the predicted target is not the detected one (BTB miss) :
            if btb_set.ways[existing_way_num].target != actual_target:
                # If confidence is 0, then replace the target :
                if btb_set.ways[existing_way_num].confidence.is_saturated_low():
                    btb_set.ways[existing_way_num].target = actual_target
                    btb_set.ways[existing_way_num].confidence.reset(self.default_confidence) # reset the confidence to 1
                # Otherwise, decrement confidence :
                else:
                    btb_set.ways[existing_way_num].confidence.decrement()
            # Otherwise if the prediction is correct (BTB hit), reinforce confidence :
            else:
                btb_set.ways[existing_way_num].confidence.increment()

            return
        
        # If entry doesn't exist, needs to be allocated :

        # Find invalid (empty) way :
        invalid_way_num = btb_set.find_invalid_way()
        if invalid_way_num is not None:
            # Found empty way -> use it :
            btb_set.ways[invalid_way_num].valid = True
            btb_set.ways[invalid_way_num].tag = tag
            btb_set.ways[invalid_way_num].target = actual_target
            btb_set.ways[invalid_way_num].confidence.reset(self.default_confidence)
            btb_set.update_lru_bits(invalid_way_num, "mru")
            return
        
        # If all ways are used, evict LRU way :
        # Find LRU way :
        self.evictions += 1 # update statistics
        LRU_way = btb_set.find_lru_way()
        # Evict and replace :
        btb_set.ways[LRU_way].valid = True
        btb_set.ways[LRU_way].tag = tag
        btb_set.ways[LRU_way].target = actual_target
        btb_set.ways[LRU_way].confidence.reset(self.default_confidence)
        btb_set.update_lru_bits(LRU_way, "mru")
    
    def flush(self):
        """
        Flush all entries from the BTB.
        (Used for context switches or security purposes)
        """

        for btb_set in self.sets:
            for entry in btb_set.ways:
                entry.valid = False
                entry.confidence.reset(self.default_confidence)
            btb_set.lru_bits = 0
    
    def flush_address(self, branch_address: int):
        """
        Flush a specific branch address from the BTB.
        
        Args :
            - branch_address [int] : address to flush
        """

        # Decompose address into set index and tag :
        set_index, tag = self._decompose_address(branch_address)

        # Get the relevant set :
        btb_set = self.sets[set_index]
        
        # Find the entry in the set :
        way_num = btb_set.find_entry(tag)
        if way_num is not None:
            # If it is found, flush it :
            btb_set.ways[way_num].valid = False
            btb_set.update_lru_bits(way_num, "rmv")
    
    def get_stats(self) -> dict:
        """
        Get BTB statistics including occupancy information.
        
        Args :
            - None
        
        Returns :
            - dict : dictionary with hit rate, miss rate, occupancy, configuration, etc.
        """

        # Compute performance stats :
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        miss_rate = self.misses / total if total > 0 else 0.0
        
        # Compute occupancy stats :
        occupied_entries = 0
        occupied_sets = 0
        full_sets = 0
        
        for btb_set in self.sets:
            set_count = 0
            for way in btb_set.ways:
                if way.valid:
                    set_count += 1
                    occupied_entries += 1
            
            if set_count > 0:
                occupied_sets += 1
            if set_count == self.associativity:
                full_sets += 1
        
        occupancy_rate = occupied_entries / self.num_entries if self.num_entries > 0 else 0.0
        
        # Return all stats :
        return {
            # Configuration parameters
            'num_sets': self.num_sets,
            'associativity': self.associativity,
            'total_capacity': self.num_entries,
            'tag_bits': self.tag_bits,
            'index_bits': self.index_bits,
            'confidence_counter_bits': self.confidence_counter_bits,
            'default_confidence': self.default_confidence,
            
            # Performance metrics
            'hits': self.hits,
            'misses': self.misses,
            'total_lookups': total,
            'hit_rate': hit_rate,
            'miss_rate': miss_rate,
            'updates': self.updates,
            'evictions': self.evictions,
            
            # Occupancy metrics
            'occupied_entries': occupied_entries,
            'occupancy_rate': occupancy_rate,
            'occupied_sets': occupied_sets,
            'full_sets': full_sets
        }


    def print_stats(self):
        """
        Print BTB statistics in a human-readable ASCII tree.
        
        Args :
            - None
        
        Returns :
            - None
        """

        # Get the stats :
        stats = self.get_stats()

        # Display them :
        print("\n┌[BTB Statistics]")
        print(f"├── Configuration")
        print(f"│   ├── Number of sets : {stats['num_sets']}")
        print(f"│   ├── Associativity : {stats['associativity']}")
        print(f"│   ├── Total capacity : {stats['total_capacity']} entries")
        print(f"│   ├── Tag bits : {stats['tag_bits']}")
        print(f"│   ├── Index bits : {stats['index_bits']}")
        print(f"│   ├── Confidence counter bits : {stats['confidence_counter_bits']}")
        print(f"│   └── Default confidence : {stats['default_confidence']}/{(1 << stats['confidence_counter_bits']) - 1}")
        print(f"├── Occupancy")
        print(f"│   ├── Occupied entries : {stats['occupied_entries']}/{stats['total_capacity']}")
        print(f"│   ├── Occupancy rate : {stats['occupancy_rate']:.2%}")
        print(f"│   ├── Occupied sets : {stats['occupied_sets']}/{stats['num_sets']}")
        print(f"│   └── Full sets : {stats['full_sets']}/{stats['num_sets']}")
        print(f"├── Hits")
        print(f"│   ├── Number of hits : {stats['hits']}")
        print(f"│   └── Hit rate : {stats['hit_rate']:.2%}")
        print(f"├── Misses")
        print(f"│   ├── Number of misses : {stats['misses']}")
        print(f"│   └── Miss rate : {stats['miss_rate']:.2%}")
        print(f"└── Number of operations")
        print(f"    ├── Lookups : {stats['total_lookups']}")
        print(f"    ├── Updates : {stats['updates']}")
        print(f"    └── Evictions : {stats['evictions']}")

# ┌─────────────────────────────┐
# │ TESTING AND DEMONSTRATION : │
# └─────────────────────────────┘

if __name__ == "__main__":

    print("╔═══════════════════╗\n║ TESTING THE BTB : ║\n╚═══════════════════╝")
    print(".")

    # Create a 4-way associative BTB with 256 entries :
    btb = BTB(num_entries=256, associativity=4)
    print(f"├── Created BTB : {btb.num_sets} sets x {btb.associativity} ways")
    print(f"└── Total capacity : {btb.num_entries} entries\n")
    
    # Test 1: Basic insertion and lookup :
    print("┌─────────────────────────────────────┐\n│ Test 1 : Basic insertion and lookup │\n└─────────────────────────────────────┘")
    print(".")
    
    branch1 = 0x401000
    target1 = 0x500000
    
    # Initially, branch should not be in BTB :
    result = btb.predict(branch1)
    print(f"├── Lookup 0x{branch1:08X} : {result} (expected None)")
    
    # Train the BTB
    btb.update(branch1, target1)
    print(f"├── Updated BTB : 0x{branch1:08X} → 0x{target1:08X}")
    
    # Now it should predict correctly
    result = btb.predict(branch1)
    print(f"└── Lookup 0x{branch1:08X} : 0x{result:08X} (expected 0x{target1:08X}) " + "[OK]\n" if result == target1 else "[FAIL]\n")
    
    # Test 2: Multiple entries in same set (XOR-folding aware)
    print("┌───────────────────────────────────────┐\n│ Test 2 : Multiple entries in same set │\n└───────────────────────────────────────┘")
    print(".")
    
    # With XOR-folding, we need to find addresses that actually map to the same set
    print("├── Searching for 4 addresses that map to the same set...")
    target_set = 15  # Arbitrary target set
    branches = []
    targets = [0x500000, 0x600000, 0x700000, 0x800000]
    search_addr = 0x400000
    
    while len(branches) < 4:
        set_idx, _ = btb._decompose_address(search_addr)
        if set_idx == target_set:
            branches.append(search_addr)
        search_addr += 4  # Next aligned address
        if search_addr > 0x800000:  # Safety limit
            break
    
    if len(branches) == 4:
        print(f"├── Found {len(branches)} addresses mapping to set {target_set}")
        print("├── Inserting all entries into the same set :")
        for i, (branch, target) in enumerate(zip(branches, targets)):
            btb.update(branch, target)
            set_idx, tag = btb._decompose_address(branch)
            prefix = "│   ├──" if i < len(branches) - 1 else "│   └──"
            print(f"{prefix} 0x{branch:08X} → 0x{target:08X} (set {set_idx}, tag 0x{tag:04X})")
        
        print("└── Verifying all entries :")
        for i, (branch, expected_target) in enumerate(zip(branches, targets)):
            result = btb.predict(branch)
            match = "[OK]" if result == expected_target else "[FAIL]"
            prefix = "    ├──" if i < len(branches) - 1 else "    └──"
            print(f"{prefix} 0x{branch:08X} → 0x{result:08X} {match}")
    else:
        print(f"└── Could only find {len(branches)} addresses (test skipped)")
    print()
    
    # Test 3: LRU eviction (XOR-folding aware)
    print("┌───────────────────────┐\n│ Test 3 : LRU eviction │\n└───────────────────────┘")
    print(".")
    
    # Create a small BTB to force evictions
    small_btb = BTB(num_entries=8, associativity=2)  # Only 4 sets × 2 ways
    print(f"├── Created small BTB : {small_btb.num_sets} sets x {small_btb.associativity} ways")
    
    # Find 3 addresses that map to the same set to force eviction
    print("├── Searching for 3 addresses that map to the same set...")
    target_set_small = 0  # Use set 0 for simplicity
    test_branches = []
    test_targets = [0xA000, 0xB000, 0xC000]
    search_addr = 0x100000
    
    while len(test_branches) < 3:
        set_idx, _ = small_btb._decompose_address(search_addr)
        if set_idx == target_set_small:
            test_branches.append(search_addr)
        search_addr += 4
        if search_addr > 0x200000:  # Safety limit
            break
    
    if len(test_branches) == 3:
        print(f"├── Found 3 addresses mapping to set {target_set_small}")
        
        # Display the addresses found
        for i, branch in enumerate(test_branches):
            set_idx, tag = small_btb._decompose_address(branch)
            prefix = "│   ├──" if i < len(test_branches) - 1 else "│   └──"
            print(f"{prefix} Address {i+1} : 0x{branch:08X} (set {set_idx}, tag 0x{tag:04X})")
        
        # Insert first two
        print("├── Inserting first two entries :")
        for i in range(2):
            small_btb.update(test_branches[i], test_targets[i])
            prefix = "│   ├──" if i == 0 else "│   └──"
            print(f"{prefix} Branch {i+1} : 0x{test_branches[i]:08X} → 0x{test_targets[i]:04X}")
        
        # Access first one to make it MRU
        small_btb.predict(test_branches[0])
        print(f"├── Accessed 0x{test_branches[0]:08X} (marking it as MRU)")
        
        # Insert third - should evict the second one (LRU)
        small_btb.update(test_branches[2], test_targets[2])
        print(f"├── Inserted branch 3 : 0x{test_branches[2]:08X} → 0x{test_targets[2]:04X}")
        print(f"├── This should evict : 0x{test_branches[1]:08X} (the LRU entry)")
        
        # Verify
        print("└── Verifying entries :")
        all_correct = True
        for i, (branch, target) in enumerate(zip(test_branches, test_targets)):
            result = small_btb.predict(branch)
            if i == 1:  # This one should be evicted
                is_correct = result is None
                match = "[OK]" if is_correct else "[FAIL]"
                all_correct = all_correct and is_correct
                prefix = "    ├──" if i < len(test_branches) - 1 else "    └──"
                print(f"{prefix} 0x{branch:08X} : {result} (expected None - evicted) {match}")
            else:
                is_correct = result == target
                match = "[OK]" if is_correct else "[FAIL]"
                all_correct = all_correct and is_correct
                prefix = "    ├──" if i < len(test_branches) - 1 else "    └──"
                print(f"{prefix} 0x{branch:08X} : 0x{result:04X} (expected 0x{target:04X}) {match}")
        
        # Show eviction statistics
        print(f"        └── Eviction count : {small_btb.evictions} (expected 1) {'[OK]' if small_btb.evictions == 1 else '[FAIL]'}")
    else:
        print(f"└── Could only find {len(test_branches)} addresses (test skipped)")
    print()
    
    # Test 4: Address decomposition and XOR-folding
    print("┌────────────────────────────────┐\n│ Test 4 : Address decomposition │\n└────────────────────────────────┘")
    print(".")
    
    test_addresses = [0x401234, 0x501234, 0x401334]
    
    print("├── Testing XOR-fold hash function :")
    for i, addr in enumerate(test_addresses):
        set_idx, tag = btb._decompose_address(addr)
        prefix = "│   ├──" if i < len(test_addresses) - 1 else "│   └──"
        print(f"{prefix} 0x{addr:08X} → set {set_idx:4d}, tag 0x{tag:04X}")
    
    # Demonstrate that similar addresses can map to different sets (anti-clustering)
    print("└── Anti-clustering property (consecutive addresses) :")
    consecutive = [0x500000, 0x500004, 0x500008, 0x50000C]
    sets_seen = set()
    for i, addr in enumerate(consecutive):
        set_idx, _ = btb._decompose_address(addr)
        sets_seen.add(set_idx)
        prefix = "    ├──" if i < len(consecutive) - 1 else "    └──"
        print(f"{prefix} 0x{addr:08X} → set {set_idx:4d}")
    print(f"        └── Spread across {len(sets_seen)} different sets (good distribution) {'[OK]' if len(sets_seen) >= 3 else '[FAIL]'}")
    print()
    
    # Test 5: Statistics
    print("┌───────────────────────────┐\n│ Test 5 : Final statistics │\n└───────────────────────────┘")
    btb.print_stats()
    print("\n")
    
    # Test 6: Collision handling and tag differentiation
    print("┌─────────────────────────────┐\n│ Test 6 : Collision handling │\n└─────────────────────────────┘")
    print(".")
    
    # Create a very small BTB to easily force collisions
    collision_btb = BTB(num_entries=4, associativity=2)  # 2 sets × 2 ways
    print(f"├── Created collision BTB : {collision_btb.num_sets} sets x {collision_btb.associativity} ways")
    
    # Find 3 addresses that map to set 0 with different tags
    print("├── Finding addresses that map to set 0 with different tags...")
    collision_set = 0
    collision_branches = []
    collision_targets = [0x1000, 0x2000, 0x3000]
    search_addr = 0x200000
    tags_seen = set()
    
    while len(collision_branches) < 3:
        set_idx, tag = collision_btb._decompose_address(search_addr)
        if set_idx == collision_set and tag not in tags_seen:
            collision_branches.append(search_addr)
            tags_seen.add(tag)
        search_addr += 4
        if search_addr > 0x400000:  # Safety limit
            break
    
    if len(collision_branches) == 3:
        print(f"├── Found 3 addresses with different tags :")
        for i, branch in enumerate(collision_branches):
            set_idx, tag = collision_btb._decompose_address(branch)
            prefix = "│   ├──" if i < len(collision_branches) - 1 else "│   └──"
            print(f"{prefix} 0x{branch:08X} → set {set_idx}, tag 0x{tag:04X}")
        
        # Insert all three (should cause one eviction due to 2-way associativity)
        print("├── Inserting all 3 entries (should evict 1) :")
        for branch, target in zip(collision_branches, collision_targets):
            collision_btb.update(branch, target)
        
        # Check how many are still present
        present_count = 0
        for branch, target in zip(collision_branches, collision_targets):
            result = collision_btb.predict(branch)
            if result == target:
                present_count += 1
        
        print(f"└── Entries present : {present_count}/3 (expected 2) {'[OK]' if present_count == 2 else '[FAIL]'}")
    else:
        print(f"└── Could only find {len(collision_branches)} addresses (test skipped)")
    print()

    # Test 7: Pseudo-LRU algorithm verification
    print("┌───────────────────────────────┐\n│ Test 7 : Pseudo-LRU algorithm │\n└───────────────────────────────┘")
    print(".")
    
    lru_btb = BTB(num_entries=4, associativity=4)  # Single set with 4 ways
    btb_set = lru_btb.sets[0]
    
    # Fill all ways
    for way in range(4):
        btb_set.ways[way].valid = True
        btb_set.ways[way].tag = way
        btb_set.ways[way].target = way * 0x1000
    
    print(f"├── Filled all 4 ways in set 0")
    print(f"├── Initial LRU bits : {btb_set.lru_bits:03b}")
    
    # Access pattern: way 3 → way 0 → way 2
    access_pattern = [3, 0, 2]
    
    print("├── Simulating access pattern : [3, 0, 2]")
    for i, way in enumerate(access_pattern):
        btb_set.update_lru_bits(way, "mru")
        lru = btb_set.find_lru_way()
        prefix = "│   ├──" if i < len(access_pattern) - 1 else "│   └──"
        print(f"{prefix} Accessed way {way} → LRU bits: {btb_set.lru_bits:03b}, LRU way: {lru}")
    
    final_lru = btb_set.find_lru_way()
    print(f"├── Way 1 was never accessed, so it should be LRU")
    print(f"└── Actual LRU : way {final_lru} {'[OK]' if final_lru == 1 else '[FAIL]'}")
    print()
    
    # Test 8: Flush operations
    print("┌───────────────────────────┐\n│ Test 8 : Flush operations │\n└───────────────────────────┘")
    print(".")
    
    flush_btb = BTB(num_entries=16, associativity=4)
    
    # Insert some entries
    flush_addresses = [0x300000, 0x300100, 0x300200]
    flush_targets = [0x400000, 0x500000, 0x600000]
    
    print("├── Inserting 3 entries :")
    for i, (addr, target) in enumerate(zip(flush_addresses, flush_targets)):
        flush_btb.update(addr, target)
        prefix = "│   ├──" if i < len(flush_addresses) - 1 else "│   └──"
        print(f"{prefix} 0x{addr:08X} → 0x{target:08X}")
    
    # Verify they're present
    print("├── Verifying entries are present :")
    all_present = True
    for addr, target in zip(flush_addresses, flush_targets):
        result = flush_btb.predict(addr)
        all_present = all_present and (result == target)
    print(f"│   └── All entries present : {all_present} {'[OK]' if all_present else '[FAIL]'}")
    
    # Flush single address
    print(f"├── Flushing address 0x{flush_addresses[1]:08X}")
    flush_btb.flush_address(flush_addresses[1])
    
    # Verify the flushed entry is gone but others remain
    print("├── Verifying selective flush :")
    result_0 = flush_btb.predict(flush_addresses[0])
    result_1 = flush_btb.predict(flush_addresses[1])
    result_2 = flush_btb.predict(flush_addresses[2])
    
    correct_selective = (result_0 == flush_targets[0] and 
                        result_1 is None and 
                        result_2 == flush_targets[2])
    print(f"│   ├── Entry 0 : {'present' if result_0 is not None else 'flushed'} {'[OK]' if result_0 is not None else '[FAIL]'}")
    print(f"│   ├── Entry 1 : {'present' if result_1 is not None else 'flushed'} {'[OK]' if result_1 is None else '[FAIL]'}")
    print(f"│   └── Entry 2 : {'present' if result_2 is not None else 'flushed'} {'[OK]' if result_2 is not None else '[FAIL]'}")
    
    # Flush all
    print("├── Flushing entire BTB")
    flush_btb.flush()
    
    # Verify all entries are gone
    all_flushed = all(flush_btb.predict(addr) is None for addr in flush_addresses)
    print(f"└── All entries flushed : {all_flushed} {'[OK]' if all_flushed else '[FAIL]'}")
    print()
    
    # Final summary
    print("╔════════════════════╗\n║ ALL TESTS COMPLETE ║\n╚════════════════════╝")