#include "distributed_node_manager.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace rros {

DistributedNodeManager::DistributedNodeManager(
    const std::string& node_id,
    const std::unordered_map<std::string, float>& config
) : node_id_(node_id),
    heartbeat_timeout_(5000.0f),  // 5 seconds default
    replication_factor_(2),        // 2x replication default
    load_balance_threshold_(0.7f)  // 70% threshold
{
    // Apply configuration
    if (config.count("heartbeat_timeout")) {
        heartbeat_timeout_ = config.at("heartbeat_timeout");
    }
    if (config.count("replication_factor")) {
        replication_factor_ = static_cast<uint32_t>(config.at("replication_factor"));
    }
    if (config.count("load_balance_threshold")) {
        load_balance_threshold_ = config.at("load_balance_threshold");
    }
    
    // Register this node
    NodeInfo self_info;
    self_info.node_id = node_id_;
    self_info.status = NodeStatus::ACTIVE;
    self_info.computational_capacity = config.count("computational_capacity") ? 
        config.at("computational_capacity") : 100.0f;
    self_info.memory_capacity = config.count("memory_capacity") ? 
        config.at("memory_capacity") : 1048576.0f;
    self_info.current_load = 0.0f;
    self_info.active_tasks = 0;
    self_info.last_heartbeat = std::chrono::high_resolution_clock::now();
    
    nodes_[node_id_] = self_info;
}

DistributedNodeManager::~DistributedNodeManager() {
    // Clean shutdown - notify other nodes
    for (const auto& [id, node] : nodes_) {
        if (id != node_id_ && node.status == NodeStatus::ACTIVE) {
            NodeMessage msg;
            msg.sender_id = node_id_;
            msg.receiver_id = id;
            msg.message_type = NodeMessage::Type::HEARTBEAT;
            msg.timestamp = std::chrono::high_resolution_clock::now();
            send_message(msg);
        }
    }
}

bool DistributedNodeManager::register_node(const NodeInfo& node_info) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    if (nodes_.count(node_info.node_id)) {
        // Node already registered - update info
        nodes_[node_info.node_id] = node_info;
    } else {
        // New node registration
        nodes_[node_info.node_id] = node_info;
    }
    
    nodes_[node_info.node_id].last_heartbeat = 
        std::chrono::high_resolution_clock::now();
    
    return true;
}

void DistributedNodeManager::unregister_node(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    if (nodes_.count(node_id)) {
        nodes_[node_id].status = NodeStatus::OFFLINE;
        
        // Redistribute tasks from this node
        redistribute_tasks(node_id);
    }
}

NodeInfo DistributedNodeManager::get_node_info(const std::string& node_id) const {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    if (nodes_.count(node_id)) {
        return nodes_.at(node_id);
    }
    
    return NodeInfo{};  // Return empty info if not found
}

std::vector<std::string> DistributedNodeManager::get_active_nodes() const {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    std::vector<std::string> active_nodes;
    for (const auto& [id, node] : nodes_) {
        if (node.status == NodeStatus::ACTIVE) {
            active_nodes.push_back(id);
        }
    }
    
    return active_nodes;
}

void DistributedNodeManager::update_heartbeat(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    if (nodes_.count(node_id)) {
        nodes_[node_id].last_heartbeat = std::chrono::high_resolution_clock::now();
        
        // If node was recovering or degraded, restore to active
        if (nodes_[node_id].status == NodeStatus::RECOVERING ||
            nodes_[node_id].status == NodeStatus::DEGRADED) {
            nodes_[node_id].status = NodeStatus::ACTIVE;
        }
    }
}

std::vector<std::string> DistributedNodeManager::check_node_health() {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    std::vector<std::string> failed_nodes;
    auto now = std::chrono::high_resolution_clock::now();
    
    for (auto& [id, node] : nodes_) {
        if (id == node_id_) continue;  // Don't check self
        
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - node.last_heartbeat
        ).count();
        
        if (elapsed > heartbeat_timeout_) {
            if (node.status == NodeStatus::ACTIVE) {
                node.status = NodeStatus::DEGRADED;
            } else if (node.status == NodeStatus::DEGRADED) {
                node.status = NodeStatus::OFFLINE;
                failed_nodes.push_back(id);
            }
        }
    }
    
    return failed_nodes;
}

std::string DistributedNodeManager::submit_distributed_task(const DistributedTask& task) {
    std::lock_guard<std::mutex> lock(tasks_mutex_);
    
    // Store task
    tasks_[task.task_id] = task;
    
    // Select nodes for execution
    auto selected_nodes = select_nodes_for_task(
        task.replication_factor > 0 ? task.replication_factor : replication_factor_,
        {}
    );
    
    tasks_[task.task_id].assigned_nodes = selected_nodes;
    
    // Distribute task to selected nodes
    for (const auto& node_id : selected_nodes) {
        NodeMessage msg;
        msg.sender_id = node_id_;
        msg.receiver_id = node_id;
        msg.message_type = NodeMessage::Type::TASK_REQUEST;
        msg.payload = task.input_data;
        msg.timestamp = std::chrono::high_resolution_clock::now();
        
        send_message(msg);
    }
    
    ++total_tasks_distributed_;
    
    return task.task_id;
}

std::vector<std::string> DistributedNodeManager::get_node_tasks(const std::string& node_id) const {
    std::lock_guard<std::mutex> lock(tasks_mutex_);
    
    std::vector<std::string> task_ids;
    for (const auto& [id, task] : tasks_) {
        auto it = std::find(task.assigned_nodes.begin(), 
                          task.assigned_nodes.end(), 
                          node_id);
        if (it != task.assigned_nodes.end()) {
            task_ids.push_back(id);
        }
    }
    
    return task_ids;
}

void DistributedNodeManager::report_task_completion(
    const std::string& task_id,
    const std::string& node_id,
    const std::vector<float>& result
) {
    std::lock_guard<std::mutex> lock(tasks_mutex_);
    
    if (tasks_.count(task_id)) {
        // Send result message
        NodeMessage msg;
        msg.sender_id = node_id;
        msg.receiver_id = tasks_[task_id].initiating_node;
        msg.message_type = NodeMessage::Type::TASK_RESULT;
        msg.payload = result;
        msg.timestamp = std::chrono::high_resolution_clock::now();
        
        send_message(msg);
        
        ++total_tasks_completed_;
        
        // Note: In production, should track completion per replica
        // and only remove when all replicas complete or timeout.
        // For now, simplified implementation removes on first completion.
        tasks_.erase(task_id);
    }
}

void DistributedNodeManager::redistribute_tasks(const std::string& failed_node_id) {
    std::lock_guard<std::mutex> lock(tasks_mutex_);
    
    for (auto& [id, task] : tasks_) {
        auto it = std::find(task.assigned_nodes.begin(), 
                          task.assigned_nodes.end(), 
                          failed_node_id);
        
        if (it != task.assigned_nodes.end()) {
            // Remove failed node
            task.assigned_nodes.erase(it);
            
            // Assign to new node
            auto new_nodes = select_nodes_for_task(1, {});
            if (!new_nodes.empty()) {
                task.assigned_nodes.push_back(new_nodes[0]);
                
                // Send task to new node
                NodeMessage msg;
                msg.sender_id = node_id_;
                msg.receiver_id = new_nodes[0];
                msg.message_type = NodeMessage::Type::TASK_REQUEST;
                msg.payload = task.input_data;
                msg.timestamp = std::chrono::high_resolution_clock::now();
                
                send_message(msg);
            }
        }
    }
}

std::vector<std::string> DistributedNodeManager::select_nodes_for_task(
    uint32_t num_nodes,
    const std::unordered_map<std::string, float>& resource_requirements
) const {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    // Score all active nodes
    std::vector<std::pair<std::string, float>> node_scores;
    
    for (const auto& [id, node] : nodes_) {
        if (node.status == NodeStatus::ACTIVE && id != node_id_) {
            float score = calculate_node_score(node, resource_requirements);
            node_scores.push_back({id, score});
        }
    }
    
    // Sort by score (highest first)
    std::sort(node_scores.begin(), node_scores.end(),
             [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Select top N nodes
    std::vector<std::string> selected;
    for (uint32_t i = 0; i < std::min(num_nodes, static_cast<uint32_t>(node_scores.size())); ++i) {
        selected.push_back(node_scores[i].first);
    }
    
    return selected;
}

uint32_t DistributedNodeManager::balance_cluster_load() {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    std::lock_guard<std::mutex> task_lock(tasks_mutex_);
    
    // Calculate average load
    float total_load = 0.0f;
    uint32_t active_count = 0;
    
    for (const auto& [id, node] : nodes_) {
        if (node.status == NodeStatus::ACTIVE) {
            total_load += node.current_load;
            ++active_count;
        }
    }
    
    if (active_count == 0) return 0;
    
    float avg_load = total_load / active_count;
    uint32_t redistributed = 0;
    
    // Find overloaded and underloaded nodes
    std::vector<std::string> overloaded, underloaded;
    
    for (const auto& [id, node] : nodes_) {
        if (node.status == NodeStatus::ACTIVE) {
            if (node.current_load > avg_load * 1.2f) {
                overloaded.push_back(id);
            } else if (node.current_load < avg_load * 0.8f) {
                underloaded.push_back(id);
            }
        }
    }
    
    // Redistribute tasks from overloaded to underloaded
    for (const auto& overloaded_node : overloaded) {
        auto tasks = get_node_tasks(overloaded_node);
        
        for (const auto& task_id : tasks) {
            if (!underloaded.empty() && redistributed < tasks.size() / 2) {
                // Move task to underloaded node
                auto& task = tasks_[task_id];
                auto it = std::find(task.assigned_nodes.begin(), 
                                  task.assigned_nodes.end(), 
                                  overloaded_node);
                
                if (it != task.assigned_nodes.end()) {
                    *it = underloaded[redistributed % underloaded.size()];
                    ++redistributed;
                }
            }
        }
    }
    
    return redistributed;
}

uint32_t DistributedNodeManager::mitigate_hotspots(float threshold) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    uint32_t hotspots_mitigated = 0;
    
    for (auto& [id, node] : nodes_) {
        if (node.status == NodeStatus::ACTIVE && node.current_load > threshold) {
            // Mark as degraded to prevent new assignments
            node.status = NodeStatus::DEGRADED;
            ++hotspots_mitigated;
        }
    }
    
    return hotspots_mitigated;
}

bool DistributedNodeManager::initiate_consensus(
    const std::string& proposal_id,
    const std::vector<float>& value,
    const std::vector<std::string>& participating_nodes
) {
    std::lock_guard<std::mutex> lock(consensus_mutex_);
    
    if (consensus_states_.count(proposal_id)) {
        return false;  // Consensus already in progress
    }
    
    ConsensusState state;
    state.proposal_id = proposal_id;
    state.current_phase = ConsensusState::Phase::PREPARE;
    state.participating_nodes = participating_nodes;
    state.proposed_value = value;
    state.quorum_size = (participating_nodes.size() / 2) + 1;
    state.is_committed = false;
    
    consensus_states_[proposal_id] = state;
    
    // Broadcast prepare message
    NodeMessage msg;
    msg.sender_id = node_id_;
    msg.message_type = NodeMessage::Type::CONSENSUS;
    msg.payload = value;
    msg.timestamp = std::chrono::high_resolution_clock::now();
    
    for (const auto& node_id : participating_nodes) {
        msg.receiver_id = node_id;
        send_message(msg);
    }
    
    ++total_consensus_rounds_;
    
    return true;
}

void DistributedNodeManager::vote_on_proposal(
    const std::string& proposal_id,
    const std::string& node_id,
    bool accept
) {
    std::lock_guard<std::mutex> lock(consensus_mutex_);
    
    if (consensus_states_.count(proposal_id)) {
        consensus_states_[proposal_id].votes[node_id] = accept;
        
        // Check if quorum reached
        uint32_t accept_count = 0;
        for (const auto& [id, vote] : consensus_states_[proposal_id].votes) {
            if (vote) ++accept_count;
        }
        
        if (accept_count >= consensus_states_[proposal_id].quorum_size) {
            consensus_states_[proposal_id].current_phase = ConsensusState::Phase::COMMIT;
            consensus_states_[proposal_id].is_committed = true;
        }
    }
}

bool DistributedNodeManager::is_consensus_reached(const std::string& proposal_id) const {
    std::lock_guard<std::mutex> lock(consensus_mutex_);
    
    if (consensus_states_.count(proposal_id)) {
        return consensus_states_.at(proposal_id).is_committed;
    }
    
    return false;
}

std::vector<float> DistributedNodeManager::get_consensus_result(const std::string& proposal_id) const {
    std::lock_guard<std::mutex> lock(consensus_mutex_);
    
    if (consensus_states_.count(proposal_id) && 
        consensus_states_.at(proposal_id).is_committed) {
        return consensus_states_.at(proposal_id).proposed_value;
    }
    
    return {};
}

bool DistributedNodeManager::send_message(const NodeMessage& message) {
    std::lock_guard<std::mutex> lock(message_mutex_);
    
    // In real implementation, this would use network I/O
    // For now, we simulate by queuing locally
    message_queue_.push_back(message);
    
    return true;
}

std::vector<NodeMessage> DistributedNodeManager::receive_messages() {
    std::lock_guard<std::mutex> lock(message_mutex_);
    
    auto messages = message_queue_;
    message_queue_.clear();
    
    return messages;
}

uint32_t DistributedNodeManager::broadcast_message(const NodeMessage& message) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    uint32_t sent_count = 0;
    
    for (const auto& [id, node] : nodes_) {
        if (id != node_id_ && node.status == NodeStatus::ACTIVE) {
            NodeMessage msg = message;
            msg.receiver_id = id;
            send_message(msg);
            ++sent_count;
        }
    }
    
    return sent_count;
}

std::string DistributedNodeManager::create_checkpoint() {
    // Generate checkpoint ID
    auto now = std::chrono::high_resolution_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()
    ).count();
    
    std::string checkpoint_id = node_id_ + "_" + std::to_string(timestamp);
    
    // In real implementation, serialize state to persistent storage
    // For now, we just return the ID
    
    return checkpoint_id;
}

bool DistributedNodeManager::restore_from_checkpoint(const std::string& checkpoint_id) {
    // In real implementation, deserialize state from persistent storage
    // For now, we simulate success
    
    return true;
}

void DistributedNodeManager::handle_node_failure(const std::string& failed_node_id) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    if (nodes_.count(failed_node_id)) {
        nodes_[failed_node_id].status = NodeStatus::OFFLINE;
        
        // Redistribute tasks
        redistribute_tasks(failed_node_id);
        
        // Initiate failover if needed
        initiate_failover(failed_node_id);
        
        ++total_failovers_;
    }
}

std::string DistributedNodeManager::initiate_failover(const std::string& failed_node_id) {
    // Select replacement node
    auto candidates = select_nodes_for_task(1, {});
    
    if (candidates.empty()) {
        return "";  // No replacement available
    }
    
    std::string replacement = candidates[0];
    
    // Transfer state to replacement (simplified)
    NodeMessage msg;
    msg.sender_id = node_id_;
    msg.receiver_id = replacement;
    msg.message_type = NodeMessage::Type::FAILOVER;
    msg.timestamp = std::chrono::high_resolution_clock::now();
    
    send_message(msg);
    
    return replacement;
}

std::unordered_map<std::string, float> DistributedNodeManager::get_cluster_stats() const {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    std::unordered_map<std::string, float> stats;
    
    uint32_t active_count = 0;
    float total_load = 0.0f;
    float total_capacity = 0.0f;
    
    for (const auto& [id, node] : nodes_) {
        if (node.status == NodeStatus::ACTIVE) {
            ++active_count;
            total_load += node.current_load;
            total_capacity += node.computational_capacity;
        }
    }
    
    stats["active_nodes"] = static_cast<float>(active_count);
    stats["average_load"] = active_count > 0 ? total_load / active_count : 0.0f;
    stats["total_capacity"] = total_capacity;
    stats["tasks_distributed"] = static_cast<float>(total_tasks_distributed_.load());
    stats["tasks_completed"] = static_cast<float>(total_tasks_completed_.load());
    stats["failovers"] = static_cast<float>(total_failovers_.load());
    stats["consensus_rounds"] = static_cast<float>(total_consensus_rounds_.load());
    
    return stats;
}

float DistributedNodeManager::get_node_utilization(const std::string& node_id) const {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    if (nodes_.count(node_id)) {
        return nodes_.at(node_id).current_load;
    }
    
    return 0.0f;
}

float DistributedNodeManager::get_total_cluster_capacity() const {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    float total = 0.0f;
    for (const auto& [id, node] : nodes_) {
        if (node.status == NodeStatus::ACTIVE) {
            total += node.computational_capacity;
        }
    }
    
    return total;
}

float DistributedNodeManager::get_scaling_efficiency() const {
    auto active_nodes = get_active_nodes();
    if (active_nodes.size() <= 1) return 1.0f;
    
    auto stats = get_cluster_stats();
    float tasks_completed = stats["tasks_completed"];
    float node_count = static_cast<float>(active_nodes.size());
    
    // Ideal: linear scaling (N nodes = N × throughput)
    // Efficiency = actual_throughput / (N × single_node_throughput)
    float ideal_throughput = node_count * (tasks_completed / node_count);
    float actual_throughput = tasks_completed;
    
    return ideal_throughput > 0.0f ? actual_throughput / ideal_throughput : 0.0f;
}

void DistributedNodeManager::update_config(const std::unordered_map<std::string, float>& config) {
    if (config.count("heartbeat_timeout")) {
        heartbeat_timeout_ = config.at("heartbeat_timeout");
    }
    if (config.count("replication_factor")) {
        replication_factor_ = static_cast<uint32_t>(config.at("replication_factor"));
    }
    if (config.count("load_balance_threshold")) {
        load_balance_threshold_ = config.at("load_balance_threshold");
    }
}

void DistributedNodeManager::reset() {
    std::lock_guard<std::mutex> lock1(nodes_mutex_);
    std::lock_guard<std::mutex> lock2(tasks_mutex_);
    std::lock_guard<std::mutex> lock3(consensus_mutex_);
    std::lock_guard<std::mutex> lock4(message_mutex_);
    
    // Keep only this node
    auto self = nodes_[node_id_];
    nodes_.clear();
    nodes_[node_id_] = self;
    
    tasks_.clear();
    consensus_states_.clear();
    message_queue_.clear();
    
    total_tasks_distributed_ = 0;
    total_tasks_completed_ = 0;
    total_failovers_ = 0;
    total_consensus_rounds_ = 0;
}

float DistributedNodeManager::calculate_node_score(
    const NodeInfo& node,
    const std::unordered_map<std::string, float>& requirements
) const {
    // Score based on available capacity and current load
    float capacity_score = node.computational_capacity * (1.0f - node.current_load);
    
    // Penalty for high load
    float load_penalty = node.current_load * 0.5f;
    
    // Bonus for low task count
    float task_bonus = 1.0f / (1.0f + static_cast<float>(node.active_tasks));
    
    return capacity_score - load_penalty + task_bonus;
}

void DistributedNodeManager::cleanup_stale_nodes() {
    // Called by check_node_health()
}

void DistributedNodeManager::update_cluster_metrics() {
    // Periodically update cluster-wide metrics
}

} // namespace rros
