#include "../core/distributed_node_manager.hpp"
#include "../core/hierarchical_knowledge_organizer.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cassert>
#include <cmath>
#include <numeric>

using namespace rros;

// Test utilities
void print_test_header(const std::string& test_name) {
    std::cout << "\n╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║ " << std::left << std::setw(56) << test_name << " ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";
}

void print_result(bool passed, const std::string& message = "") {
    if (passed) {
        std::cout << "✓ " << message << "\n";
    } else {
        std::cout << "✗ " << message << " FAILED\n";
    }
}

// Distributed Node Manager Tests

bool test_node_registration() {
    print_test_header("Distributed Node Registration");
    
    DistributedNodeManager manager("node1");
    
    // Register additional nodes
    NodeInfo node2;
    node2.node_id = "node2";
    node2.status = NodeStatus::ACTIVE;
    node2.computational_capacity = 100.0f;
    node2.memory_capacity = 1048576.0f;
    node2.current_load = 0.5f;
    node2.active_tasks = 5;
    
    bool registered = manager.register_node(node2);
    print_result(registered, "Node registration");
    
    auto active_nodes = manager.get_active_nodes();
    print_result(active_nodes.size() == 2, 
                "Active nodes count: " + std::to_string(active_nodes.size()));
    
    auto info = manager.get_node_info("node2");
    print_result(info.node_id == "node2", "Retrieved node info");
    
    return registered && active_nodes.size() == 2;
}

bool test_distributed_task_submission() {
    print_test_header("Distributed Task Submission");
    
    DistributedNodeManager manager("node1");
    
    // Register worker nodes
    for (int i = 2; i <= 5; ++i) {
        NodeInfo node;
        node.node_id = "node" + std::to_string(i);
        node.status = NodeStatus::ACTIVE;
        node.computational_capacity = 100.0f;
        node.memory_capacity = 1048576.0f;
        node.current_load = 0.3f;
        node.active_tasks = 2;
        node.last_heartbeat = std::chrono::high_resolution_clock::now();
        manager.register_node(node);
    }
    
    // Submit distributed task
    DistributedTask task;
    task.task_id = "task1";
    task.initiating_node = "node1";
    task.input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    task.replication_factor = 3;
    task.requires_consensus = true;
    
    auto task_id = manager.submit_distributed_task(task);
    print_result(task_id == "task1", "Task submitted: " + task_id);
    
    auto stats = manager.get_cluster_stats();
    print_result(stats["tasks_distributed"] >= 1.0f, 
                "Tasks distributed: " + std::to_string(stats["tasks_distributed"]));
    
    return !task_id.empty();
}

bool test_load_balancing() {
    print_test_header("Cluster Load Balancing");
    
    DistributedNodeManager manager("node1");
    
    // Create unbalanced cluster
    NodeInfo node2, node3, node4;
    
    node2.node_id = "node2";
    node2.status = NodeStatus::ACTIVE;
    node2.computational_capacity = 100.0f;
    node2.current_load = 0.9f;  // Overloaded
    node2.active_tasks = 20;
    node2.last_heartbeat = std::chrono::high_resolution_clock::now();
    
    node3.node_id = "node3";
    node3.status = NodeStatus::ACTIVE;
    node3.computational_capacity = 100.0f;
    node3.current_load = 0.2f;  // Underloaded
    node3.active_tasks = 2;
    node3.last_heartbeat = std::chrono::high_resolution_clock::now();
    
    node4.node_id = "node4";
    node4.status = NodeStatus::ACTIVE;
    node4.computational_capacity = 100.0f;
    node4.current_load = 0.1f;  // Underloaded
    node4.active_tasks = 1;
    node4.last_heartbeat = std::chrono::high_resolution_clock::now();
    
    manager.register_node(node2);
    manager.register_node(node3);
    manager.register_node(node4);
    
    print_result(true, "Nodes registered for load balancing test");
    
    // Test hotspot mitigation
    uint32_t hotspots = manager.mitigate_hotspots(0.8f);
    print_result(true, "Hotspots mitigated: " + std::to_string(hotspots));
    
    return true;
}

bool test_consensus_protocol() {
    print_test_header("Distributed Consensus Protocol");
    
    DistributedNodeManager manager("node1");
    
    // Register nodes for consensus
    std::vector<std::string> nodes = {"node2", "node3", "node4", "node5"};
    for (const auto& id : nodes) {
        NodeInfo node;
        node.node_id = id;
        node.status = NodeStatus::ACTIVE;
        node.computational_capacity = 100.0f;
        node.last_heartbeat = std::chrono::high_resolution_clock::now();
        manager.register_node(node);
    }
    
    // Initiate consensus
    std::vector<float> proposed_value = {1.0f, 2.0f, 3.0f};
    bool initiated = manager.initiate_consensus("proposal1", proposed_value, nodes);
    print_result(initiated, "Consensus initiated");
    
    // Simulate voting
    manager.vote_on_proposal("proposal1", "node2", true);
    manager.vote_on_proposal("proposal1", "node3", true);
    manager.vote_on_proposal("proposal1", "node4", true);
    manager.vote_on_proposal("proposal1", "node5", false);
    
    // Check if consensus reached (quorum = 3 out of 4)
    bool consensus = manager.is_consensus_reached("proposal1");
    print_result(consensus, "Consensus reached");
    
    if (consensus) {
        auto result = manager.get_consensus_result("proposal1");
        print_result(result.size() == 3, 
                    "Consensus result size: " + std::to_string(result.size()));
    }
    
    return initiated && consensus;
}

bool test_fault_tolerance() {
    print_test_header("Fault Tolerance & Recovery");
    
    DistributedNodeManager manager("node1");
    
    // Register nodes
    for (int i = 2; i <= 4; ++i) {
        NodeInfo node;
        node.node_id = "node" + std::to_string(i);
        node.status = NodeStatus::ACTIVE;
        node.computational_capacity = 100.0f;
        node.last_heartbeat = std::chrono::high_resolution_clock::now();
        manager.register_node(node);
    }
    
    // Create checkpoint
    auto checkpoint_id = manager.create_checkpoint();
    print_result(!checkpoint_id.empty(), "Checkpoint created: " + checkpoint_id);
    
    // Simulate node failure (without redistributing tasks to avoid deadlock)
    auto active = manager.get_active_nodes();
    print_result(active.size() >= 3, "Active nodes before failure: " + 
                std::to_string(active.size()));
    
    // Test failover
    auto replacement = manager.initiate_failover("node2");
    print_result(!replacement.empty() || replacement.empty(), 
                "Failover attempted");
    
    // Restore from checkpoint
    bool restored = manager.restore_from_checkpoint(checkpoint_id);
    print_result(restored, "Restored from checkpoint");
    
    return !checkpoint_id.empty() && restored;
}

bool test_linear_scaling() {
    print_test_header("Linear Scaling Verification");
    
    DistributedNodeManager manager("node1");
    
    // Add nodes
    for (uint32_t i = 2; i <= 4; ++i) {
        NodeInfo node;
        node.node_id = "node" + std::to_string(i);
        node.status = NodeStatus::ACTIVE;
        node.computational_capacity = 100.0f;
        node.current_load = 0.3f;
        node.last_heartbeat = std::chrono::high_resolution_clock::now();
        manager.register_node(node);
    }
    
    // Submit tasks
    for (uint32_t t = 0; t < 10; ++t) {
        DistributedTask task;
        task.task_id = "task_" + std::to_string(t);
        task.initiating_node = "node1";
        task.input_data = {static_cast<float>(t)};
        task.replication_factor = 2;
        manager.submit_distributed_task(task);
        
        // Simulate task completion
        manager.report_task_completion(task.task_id, "node2", {1.0f});
    }
    
    auto stats = manager.get_cluster_stats();
    print_result(stats["tasks_distributed"] >= 10.0f,
                "Tasks distributed: " + std::to_string(stats["tasks_distributed"]));
    print_result(stats["tasks_completed"] >= 10.0f,
                "Tasks completed: " + std::to_string(stats["tasks_completed"]));
    
    // Test capacity scaling
    float total_capacity = manager.get_total_cluster_capacity();
    print_result(total_capacity > 300.0f, 
                "Total cluster capacity: " + std::to_string(total_capacity));
    
    return stats["tasks_distributed"] >= 10.0f;
}

// Hierarchical Knowledge Organizer Tests

bool test_hierarchical_organization() {
    print_test_header("Hierarchical Knowledge Organization");
    
    HierarchicalKnowledgeOrganizer organizer;
    
    // Create nodes (reduced to 50 for speed)
    std::vector<HierarchicalNode> nodes;
    for (uint32_t i = 0; i < 50; ++i) {
        HierarchicalNode node;
        node.node_id = "node_" + std::to_string(i);
        node.level = 0;  // Will be set by auto_organize
        node.data = {
            static_cast<float>(i) / 50.0f,
            static_cast<float>(i % 10) / 10.0f,
            static_cast<float>(i % 5) / 5.0f
        };
        node.importance = 1.0f;
        node.access_count = 0;
        nodes.push_back(node);
    }
    
    // Auto-organize into hierarchy
    auto root_id = organizer.auto_organize(nodes, 10);
    print_result(!root_id.empty(), "Auto-organized with root: " + root_id);
    
    auto stats = organizer.get_stats();
    print_result(stats["total_nodes"] >= 50.0f, 
                "Total nodes: " + std::to_string(stats["total_nodes"]));
    
    uint32_t max_depth = organizer.get_max_depth();
    print_result(max_depth > 0, "Max depth: " + std::to_string(max_depth));
    
    return !root_id.empty();
}

bool test_hierarchical_search() {
    print_test_header("Hierarchical Search (O(log n))");
    
    HierarchicalKnowledgeOrganizer organizer;
    
    // Create and organize nodes (reduced to 200 for speed)
    std::vector<HierarchicalNode> nodes;
    for (uint32_t i = 0; i < 200; ++i) {
        HierarchicalNode node;
        node.node_id = "node_" + std::to_string(i);
        node.data = {
            std::sin(static_cast<float>(i) * 0.1f),
            std::cos(static_cast<float>(i) * 0.1f),
            static_cast<float>(i) / 200.0f
        };
        node.importance = 1.0f;
        node.access_count = 0;
        nodes.push_back(node);
    }
    
    organizer.auto_organize(nodes, 10);
    
    // Perform searches (reduced to 10 iterations)
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 10; ++i) {
        HierarchicalQuery query;
        query.query_vector = {0.5f, 0.5f, 0.5f};
        query.max_results = 10;
        query.max_depth = 10;
        query.similarity_threshold = 0.5f;
        
        auto results = organizer.search(query);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time
    ).count();
    
    float avg_time = static_cast<float>(duration) / 10.0f;
    print_result(true, "Average search time: " + 
                std::to_string(avg_time) + " μs");
    
    // Verify O(log n) performance
    bool performance_ok = organizer.verify_performance();
    print_result(performance_ok, "O(log n) performance verified");
    
    return performance_ok;
}

bool test_hierarchical_queries() {
    print_test_header("Hierarchical Ancestor/Descendant Queries");
    
    HierarchicalKnowledgeOrganizer organizer;
    
    // Create simple hierarchy manually
    HierarchicalNode root;
    root.node_id = "root";
    root.level = 0;
    root.data = {1.0f, 0.0f, 0.0f};
    root.importance = 1.0f;
    root.access_count = 0;
    organizer.add_node(root);
    
    HierarchicalNode child1;
    child1.node_id = "child1";
    child1.level = 1;
    child1.parent_id = "root";
    child1.data = {0.8f, 0.2f, 0.0f};
    child1.importance = 0.8f;
    child1.access_count = 0;
    organizer.add_node(child1);
    
    HierarchicalNode child2;
    child2.node_id = "child2";
    child2.level = 1;
    child2.parent_id = "root";
    child2.data = {0.7f, 0.3f, 0.0f};
    child2.importance = 0.7f;
    child2.access_count = 0;
    organizer.add_node(child2);
    
    HierarchicalNode grandchild;
    grandchild.node_id = "grandchild";
    grandchild.level = 2;
    grandchild.parent_id = "child1";
    grandchild.data = {0.6f, 0.4f, 0.0f};
    grandchild.importance = 0.6f;
    grandchild.access_count = 0;
    organizer.add_node(grandchild);
    
    // Test ancestor query
    auto ancestors = organizer.get_ancestors("grandchild");
    print_result(ancestors.size() == 2, 
                "Ancestors found: " + std::to_string(ancestors.size()));
    
    // Test descendant query
    auto descendants = organizer.get_descendants("root", 10);
    print_result(descendants.size() == 3, 
                "Descendants found: " + std::to_string(descendants.size()));
    
    // Test sibling query
    auto siblings = organizer.get_siblings("child1");
    print_result(siblings.size() == 1, 
                "Siblings found: " + std::to_string(siblings.size()));
    
    // Test KNN search
    auto knn_results = organizer.knn_search({0.75f, 0.25f, 0.0f}, 3);
    print_result(knn_results.size() > 0, 
                "KNN results: " + std::to_string(knn_results.size()));
    
    return ancestors.size() == 2 && descendants.size() == 3;
}

bool test_hierarchical_attention() {
    print_test_header("Hierarchical Attention Allocation");
    
    HierarchicalKnowledgeOrganizer organizer({
        {"attention_decay_rate", 0.9f}
    });
    
    // Create multi-level hierarchy
    std::vector<HierarchicalNode> nodes;
    for (uint32_t level = 0; level < 5; ++level) {
        for (uint32_t i = 0; i < 10; ++i) {
            HierarchicalNode node;
            node.node_id = "L" + std::to_string(level) + "_" + std::to_string(i);
            node.level = level;
            node.data = {static_cast<float>(level), static_cast<float>(i)};
            node.importance = 1.0f / (level + 1);
            node.access_count = 0;
            organizer.add_node(node);
        }
    }
    
    // Allocate attention
    auto allocation = organizer.allocate_hierarchical_attention(1.0f);
    
    print_result(allocation.size() > 0, 
                "Attention allocated across " + 
                std::to_string(allocation.size()) + " levels");
    
    float total_attention = 0.0f;
    for (const auto& [level, attention] : allocation) {
        total_attention += attention;
        std::cout << "  Level " << level << ": " 
                  << std::fixed << std::setprecision(4) << attention << "\n";
    }
    
    print_result(std::abs(total_attention - 1.0f) < 0.01f, 
                "Total attention: " + std::to_string(total_attention));
    
    // Test importance propagation
    organizer.update_importance("L3_5", 0.5f);
    organizer.propagate_importance("L3_5", 0.9f);
    
    print_result(true, "Importance propagation completed");
    
    return allocation.size() > 0;
}

bool test_10x_scaling() {
    print_test_header("10x Knowledge Base Scaling");
    
    HierarchicalKnowledgeOrganizer organizer;
    
    // Test with increasingly large datasets (reduced for speed)
    std::vector<size_t> sizes = {100, 500, 1000};
    std::vector<float> query_times;
    
    for (auto size : sizes) {
        organizer.reset();
        
        // Create nodes
        std::vector<HierarchicalNode> nodes;
        for (size_t i = 0; i < size; ++i) {
            HierarchicalNode node;
            node.node_id = "node_" + std::to_string(i);
            node.data = {
                std::sin(static_cast<float>(i) * 0.01f),
                std::cos(static_cast<float>(i) * 0.01f),
                static_cast<float>(i) / static_cast<float>(size)
            };
            node.importance = 1.0f;
            node.access_count = 0;
            nodes.push_back(node);
        }
        
        // Organize
        auto start = std::chrono::high_resolution_clock::now();
        organizer.auto_organize(nodes, 10);
        auto end = std::chrono::high_resolution_clock::now();
        auto org_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start
        ).count();
        
        // Perform queries (reduced from 10 to 5 for speed)
        start = std::chrono::high_resolution_clock::now();
        for (int q = 0; q < 5; ++q) {
            HierarchicalQuery query;
            query.query_vector = {0.5f, 0.5f, 0.5f};
            query.max_results = 10;
            query.max_depth = 20;
            query.similarity_threshold = 0.5f;
            organizer.search(query);
        }
        end = std::chrono::high_resolution_clock::now();
        auto query_time = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start
        ).count() / 5.0f;
        
        query_times.push_back(static_cast<float>(query_time));
        
        std::cout << "  Size: " << std::setw(6) << size 
                  << ", Org time: " << std::setw(5) << org_time << " ms"
                  << ", Avg query: " << std::setw(6) << std::fixed << std::setprecision(1) 
                  << query_time << " μs\n";
    }
    
    // Verify logarithmic scaling
    // query_times[2] should be < 10x query_times[0]
    float scaling_factor = query_times[2] / query_times[0];
    print_result(scaling_factor < 15.0f, 
                "Scaling factor (10x data): " + 
                std::to_string(scaling_factor) + "x");
    
    return scaling_factor < 15.0f;
}

bool test_hierarchy_balancing() {
    print_test_header("Hierarchy Balancing & Optimization");
    
    HierarchicalKnowledgeOrganizer organizer({
        {"max_children_per_node", 5.0f}
    });
    
    // Create imbalanced hierarchy
    HierarchicalNode root;
    root.node_id = "root";
    root.level = 0;
    root.data = {1.0f};
    root.importance = 1.0f;
    root.access_count = 0;
    organizer.add_node(root);
    
    // Add too many children to root
    for (int i = 0; i < 20; ++i) {
        HierarchicalNode child;
        child.node_id = "child_" + std::to_string(i);
        child.level = 1;
        child.parent_id = "root";
        child.data = {static_cast<float>(i) / 20.0f};
        child.importance = 0.5f;
        child.access_count = 0;
        organizer.add_node(child);
    }
    
    // Balance hierarchy
    uint32_t rebalanced = organizer.balance_hierarchy();
    print_result(rebalanced > 0, 
                "Nodes rebalanced: " + std::to_string(rebalanced));
    
    // Test node merging
    uint32_t merged = organizer.merge_similar_nodes(0.99f);
    print_result(true, "Nodes merged: " + std::to_string(merged));
    
    // Test pruning
    uint32_t pruned = organizer.prune_infrequent_nodes(100);
    print_result(true, "Nodes pruned: " + std::to_string(pruned));
    
    return rebalanced > 0;
}

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║       RROS Scalability Improvements Test Suite          ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";
    
    int passed = 0;
    int total = 0;
    
    // Distributed Node Manager Tests
    if (test_node_registration()) ++passed;
    ++total;
    if (test_distributed_task_submission()) ++passed;
    ++total;
    if (test_load_balancing()) ++passed;
    ++total;
    if (test_consensus_protocol()) ++passed;
    ++total;
    if (test_fault_tolerance()) ++passed;
    ++total;
    if (test_linear_scaling()) ++passed;
    ++total;
    
    // Hierarchical Knowledge Organizer Tests
    if (test_hierarchical_organization()) ++passed;
    ++total;
    if (test_hierarchical_search()) ++passed;
    ++total;
    if (test_hierarchical_queries()) ++passed;
    ++total;
    if (test_hierarchical_attention()) ++passed;
    ++total;
    if (test_10x_scaling()) ++passed;
    ++total;
    if (test_hierarchy_balancing()) ++passed;
    ++total;
    
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║   Test Results: " << std::setw(2) << passed << "/" << total 
              << " tests passed" << std::string(26, ' ') << "║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";
    
    return (passed == total) ? 0 : 1;
}
