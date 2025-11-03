#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <mutex>
#include <atomic>
#include <chrono>
#include <functional>

namespace rros {

/**
 * @brief Node status in distributed system
 */
enum class NodeStatus {
    ACTIVE,           // Node is active and processing
    IDLE,            // Node is available but idle
    DEGRADED,        // Node is partially functional
    OFFLINE,         // Node is offline/unreachable
    RECOVERING       // Node is recovering from failure
};

/**
 * @brief Distributed node information
 */
struct NodeInfo {
    std::string node_id;                           // Unique node identifier
    NodeStatus status;                             // Current node status
    float computational_capacity;                   // Total compute capacity
    float memory_capacity;                         // Total memory capacity
    float current_load;                            // Current processing load [0.0, 1.0]
    uint32_t active_tasks;                         // Number of active tasks
    std::chrono::high_resolution_clock::time_point last_heartbeat;
    std::unordered_map<std::string, float> resource_usage;  // Resource utilization
};

/**
 * @brief Message for inter-node communication
 */
struct NodeMessage {
    enum class Type {
        HEARTBEAT,        // Node health check
        TASK_REQUEST,     // Request task execution
        TASK_RESULT,      // Task execution result
        RESOURCE_SYNC,    // Resource synchronization
        ATTENTION_SYNC,   // Attention allocation sync
        MEMORY_SYNC,      // Memory coherence sync
        CONSENSUS,        // Consensus protocol message
        FAILOVER          // Failover coordination
    };
    
    std::string sender_id;
    std::string receiver_id;
    Type message_type;
    std::vector<float> payload;
    std::chrono::high_resolution_clock::time_point timestamp;
    uint64_t sequence_number;
};

/**
 * @brief Distributed task for cross-node execution
 */
struct DistributedTask {
    std::string task_id;
    std::string initiating_node;
    std::vector<std::string> assigned_nodes;      // Nodes assigned to task
    std::function<std::vector<float>(const std::vector<float>&)> task_function;
    std::vector<float> input_data;
    bool requires_consensus;                       // Whether task needs consensus
    uint32_t replication_factor;                  // How many nodes should execute
    std::chrono::milliseconds deadline;
};

/**
 * @brief Consensus protocol state
 */
struct ConsensusState {
    enum class Phase {
        PREPARE,          // Prepare phase
        PROPOSE,          // Proposal phase  
        ACCEPT,           // Accept phase
        COMMIT            // Commit phase
    };
    
    std::string proposal_id;
    Phase current_phase;
    std::vector<std::string> participating_nodes;
    std::unordered_map<std::string, bool> votes;  // Node votes
    std::vector<float> proposed_value;
    uint32_t quorum_size;                         // Required votes
    bool is_committed;
};

/**
 * @brief Distributed Node Manager
 * 
 * Manages distributed cognitive processing across multiple nodes.
 * Implements node discovery, health monitoring, task distribution,
 * consensus protocols, and fault tolerance mechanisms.
 */
class DistributedNodeManager {
public:
    /**
     * @brief Constructor initializes distributed node manager
     * @param node_id Unique identifier for this node
     * @param config Configuration parameters
     */
    explicit DistributedNodeManager(
        const std::string& node_id,
        const std::unordered_map<std::string, float>& config = {}
    );
    
    /**
     * @brief Destructor ensures clean shutdown
     */
    ~DistributedNodeManager();
    
    // Node Management
    
    /**
     * @brief Register a new node in the distributed system
     * @param node_info Node information
     * @return True if registration successful
     */
    bool register_node(const NodeInfo& node_info);
    
    /**
     * @brief Unregister a node from the system
     * @param node_id Node identifier
     */
    void unregister_node(const std::string& node_id);
    
    /**
     * @brief Get information about a specific node
     * @param node_id Node identifier
     * @return Node information
     */
    NodeInfo get_node_info(const std::string& node_id) const;
    
    /**
     * @brief Get all active nodes in the system
     * @return Vector of active node IDs
     */
    std::vector<std::string> get_active_nodes() const;
    
    /**
     * @brief Update node heartbeat
     * @param node_id Node identifier
     */
    void update_heartbeat(const std::string& node_id);
    
    /**
     * @brief Check node health and detect failures
     * @return Vector of failed node IDs
     */
    std::vector<std::string> check_node_health();
    
    // Task Distribution
    
    /**
     * @brief Submit distributed task for execution
     * @param task Distributed task
     * @return Task ID
     */
    std::string submit_distributed_task(const DistributedTask& task);
    
    /**
     * @brief Get task assignment for a specific node
     * @param node_id Node identifier
     * @return Vector of assigned task IDs
     */
    std::vector<std::string> get_node_tasks(const std::string& node_id) const;
    
    /**
     * @brief Report task completion
     * @param task_id Task identifier
     * @param node_id Node that completed task
     * @param result Task result
     */
    void report_task_completion(
        const std::string& task_id,
        const std::string& node_id,
        const std::vector<float>& result
    );
    
    /**
     * @brief Redistribute tasks after node failure
     * @param failed_node_id Failed node identifier
     */
    void redistribute_tasks(const std::string& failed_node_id);
    
    // Load Balancing
    
    /**
     * @brief Select optimal nodes for task execution
     * @param num_nodes Number of nodes needed
     * @param resource_requirements Resource requirements
     * @return Selected node IDs
     */
    std::vector<std::string> select_nodes_for_task(
        uint32_t num_nodes,
        const std::unordered_map<std::string, float>& resource_requirements
    ) const;
    
    /**
     * @brief Balance load across all nodes
     * @return Number of tasks redistributed
     */
    uint32_t balance_cluster_load();
    
    /**
     * @brief Detect and mitigate computational hotspots
     * @param threshold Load threshold for hotspot detection
     * @return Number of hotspots mitigated
     */
    uint32_t mitigate_hotspots(float threshold = 0.8f);
    
    // Consensus Protocol
    
    /**
     * @brief Initiate consensus protocol
     * @param proposal_id Unique proposal identifier
     * @param value Value to reach consensus on
     * @param participating_nodes Nodes in consensus
     * @return True if consensus initiated
     */
    bool initiate_consensus(
        const std::string& proposal_id,
        const std::vector<float>& value,
        const std::vector<std::string>& participating_nodes
    );
    
    /**
     * @brief Vote on consensus proposal
     * @param proposal_id Proposal identifier
     * @param node_id Voting node
     * @param accept Accept or reject proposal
     */
    void vote_on_proposal(
        const std::string& proposal_id,
        const std::string& node_id,
        bool accept
    );
    
    /**
     * @brief Check if consensus is reached
     * @param proposal_id Proposal identifier
     * @return True if consensus reached
     */
    bool is_consensus_reached(const std::string& proposal_id) const;
    
    /**
     * @brief Get consensus result
     * @param proposal_id Proposal identifier
     * @return Consensus value
     */
    std::vector<float> get_consensus_result(const std::string& proposal_id) const;
    
    // Communication
    
    /**
     * @brief Send message to another node
     * @param message Message to send
     * @return True if message sent successfully
     */
    bool send_message(const NodeMessage& message);
    
    /**
     * @brief Receive pending messages
     * @return Vector of received messages
     */
    std::vector<NodeMessage> receive_messages();
    
    /**
     * @brief Broadcast message to all nodes
     * @param message Message to broadcast
     * @return Number of nodes reached
     */
    uint32_t broadcast_message(const NodeMessage& message);
    
    // Fault Tolerance
    
    /**
     * @brief Create checkpoint of distributed state
     * @return Checkpoint ID
     */
    std::string create_checkpoint();
    
    /**
     * @brief Restore from checkpoint
     * @param checkpoint_id Checkpoint identifier
     * @return True if restore successful
     */
    bool restore_from_checkpoint(const std::string& checkpoint_id);
    
    /**
     * @brief Handle node failure
     * @param failed_node_id Failed node identifier
     */
    void handle_node_failure(const std::string& failed_node_id);
    
    /**
     * @brief Initiate automatic failover
     * @param failed_node_id Failed node identifier
     * @return Replacement node ID
     */
    std::string initiate_failover(const std::string& failed_node_id);
    
    // Monitoring
    
    /**
     * @brief Get cluster-wide statistics
     * @return Statistics map
     */
    std::unordered_map<std::string, float> get_cluster_stats() const;
    
    /**
     * @brief Get node utilization
     * @param node_id Node identifier
     * @return Utilization [0.0, 1.0]
     */
    float get_node_utilization(const std::string& node_id) const;
    
    /**
     * @brief Get total cluster capacity
     * @return Total capacity across all nodes
     */
    float get_total_cluster_capacity() const;
    
    /**
     * @brief Get scaling efficiency metric
     * @return Efficiency score [0.0, 1.0]
     */
    float get_scaling_efficiency() const;
    
    // Configuration
    
    /**
     * @brief Update configuration
     * @param config New configuration parameters
     */
    void update_config(const std::unordered_map<std::string, float>& config);
    
    /**
     * @brief Reset distributed node manager state
     */
    void reset();

private:
    std::string node_id_;                          // This node's ID
    std::unordered_map<std::string, NodeInfo> nodes_;  // All registered nodes
    std::unordered_map<std::string, DistributedTask> tasks_;  // Active tasks
    std::unordered_map<std::string, ConsensusState> consensus_states_;  // Active consensus
    std::vector<NodeMessage> message_queue_;       // Incoming messages
    
    // Configuration
    float heartbeat_timeout_;                      // Timeout for node heartbeat
    uint32_t replication_factor_;                  // Default replication factor
    float load_balance_threshold_;                 // Threshold for rebalancing
    
    // Synchronization
    mutable std::mutex nodes_mutex_;
    mutable std::mutex tasks_mutex_;
    mutable std::mutex consensus_mutex_;
    mutable std::mutex message_mutex_;
    
    // Metrics
    std::atomic<uint64_t> total_tasks_distributed_{0};
    std::atomic<uint64_t> total_tasks_completed_{0};
    std::atomic<uint64_t> total_failovers_{0};
    std::atomic<uint64_t> total_consensus_rounds_{0};
    
    // Helper methods
    float calculate_node_score(const NodeInfo& node, 
                              const std::unordered_map<std::string, float>& requirements) const;
    void cleanup_stale_nodes();
    void update_cluster_metrics();
};

} // namespace rros
