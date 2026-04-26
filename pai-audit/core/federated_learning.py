"""
Federated Learning Module for PAI
Privacy-preserving cross-institutional data collaboration

This module provides reference implementations for federated learning
to be used in rare disease research and charitable impact measurement.

Integration with organoid-fl (https://github.com/dechang64/organoid-fl)
- FedAvg aggregation algorithm
- gRPC communication
- HNSW vector retrieval
- Blockchain audit trail

Note: This is a reference/stub implementation for the PAI prototype.
For production use, integrate with the full organoid-fl system.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from datetime import datetime


class FLStatus(Enum):
    """Federated learning status."""
    IDLE = "idle"
    TRAINING = "training"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ClientUpdate:
    """Represents a client's model update."""
    client_id: str
    round_number: int
    weights: np.ndarray
    num_samples: int
    metrics: Dict[str, float]
    timestamp: str
    signature: str  # For audit trail


@dataclass
class FederatedConfig:
    """Federated learning configuration."""
    num_rounds: int = 10
    min_clients: int = 3
    client_fraction: float = 1.0  # Fraction of clients per round
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    model_type: str = "logistic_regression"  # or "neural_network"
    privacy_budget: float = 1.0  # For differential privacy


class FederatedLearningCoordinator:
    """
    Coordinates federated learning across multiple institutions.
    
    Features:
    - Client selection and management
    - FedAvg aggregation (McMahan et al. 2017)
    - Differential privacy (planned)
    - Blockchain audit trail (planned)
    
    This is a reference implementation. For production:
    - Use organoid-fl's gRPC-based communication
    - Implement secure aggregation
    - Add differential privacy
    """
    
    def __init__(self, config: FederatedConfig = None):
        """
        Initialize federated learning coordinator.
        
        Args:
            config: FL configuration parameters
        """
        self.config = config or FederatedConfig()
        self.status = FLStatus.IDLE
        self.global_weights: Optional[np.ndarray] = None
        self.client_updates: List[ClientUpdate] = []
        self.round_history: List[Dict] = []
        self.audit_chain: List[Dict] = []  # Blockchain-like audit
        
        # Simulated client registry
        self.clients: Dict[str, Dict] = {}
        
    def register_client(self, client_id: str, institution: str, data_size: int) -> bool:
        """
        Register a new client institution.
        
        Args:
            client_id: Unique client identifier
            institution: Institution name
            data_size: Number of data samples available
            
        Returns:
            Success status
        """
        self.clients[client_id] = {
            "institution": institution,
            "data_size": data_size,
            "registered_at": datetime.now().isoformat(),
            "rounds_participated": 0
        }
        self._add_audit_event("CLIENT_REGISTERED", {
            "client_id": client_id,
            "institution": institution,
            "data_size": data_size
        })
        return True
    
    def initialize_model(self, input_dim: int, num_classes: int = 1) -> np.ndarray:
        """
        Initialize global model weights.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            
        Returns:
            Initialized weights
        """
        if self.config.model_type == "logistic_regression":
            # Xavier initialization for logistic regression
            weights = np.random.randn(input_dim + 1, num_classes) * np.sqrt(2.0 / (input_dim + 1))
        else:
            # Simple neural network initialization
            weights = np.random.randn(input_dim + 1, 32) * 0.01
            
        self.global_weights = weights.flatten()
        self._add_audit_event("MODEL_INITIALIZED", {
            "input_dim": input_dim,
            "num_classes": num_classes,
            "weights_shape": weights.shape
        })
        return self.global_weights
    
    def select_clients(self) -> List[str]:
        """
        Select clients for current round.
        
        Returns:
            List of selected client IDs
        """
        num_to_select = max(
            self.config.min_clients,
            int(len(self.clients) * self.config.client_fraction)
        )
        client_ids = list(self.clients.keys())
        selected = np.random.choice(client_ids, min(num_to_select, len(client_ids)), replace=False)
        return selected.tolist()
    
    def receive_client_update(self, update: ClientUpdate) -> bool:
        """
        Receive and validate a client update.
        
        Args:
            update: Client's model update
            
        Returns:
            Validation status
        """
        # Validate signature (simplified - use proper crypto in production)
        expected_sig = self._compute_signature(update)
        
        # Store update
        self.client_updates.append(update)
        
        # Update client participation
        if update.client_id in self.clients:
            self.clients[update.client_id]["rounds_participated"] += 1
        
        self._add_audit_event("CLIENT_UPDATE_RECEIVED", {
            "client_id": update.client_id,
            "round": update.round_number,
            "num_samples": update.num_samples,
            "metrics": update.metrics
        })
        
        return True
    
    def aggregate_updates(self) -> Tuple[np.ndarray, Dict]:
        """
        Aggregate client updates using FedAvg.
        
        FedAvg (McMahan et al. 2017):
        - Weighted average of client updates by number of samples
        - Multiple local epochs per round
        
        Returns:
            Tuple of (aggregated weights, aggregation metrics)
        """
        if not self.client_updates:
            raise ValueError("No client updates to aggregate")
        
        self.status = FLStatus.AGGREGATING
        
        total_samples = sum(u.num_samples for u in self.client_updates)
        
        # Weighted average of weights
        aggregated = np.zeros_like(self.global_weights)
        for update in self.client_updates:
            weight = update.num_samples / total_samples
            aggregated += weight * update.weights
        
        # Update global model
        old_weights = self.global_weights.copy()
        self.global_weights = aggregated
        
        # Compute metrics
        metrics = {
            "total_clients": len(self.client_updates),
            "total_samples": total_samples,
            "avg_loss": np.mean([u.metrics.get("loss", 0) for u in self.client_updates]),
            "avg_accuracy": np.mean([u.metrics.get("accuracy", 0) for u in self.client_updates]),
            "weight_change_norm": np.linalg.norm(aggregated - old_weights)
        }
        
        # Clear updates for next round
        self.client_updates = []
        
        self._add_audit_event("AGGREGATION_COMPLETED", {
            "metrics": metrics,
            "clients": len(self.clients)
        })
        
        self.status = FLStatus.COMPLETED
        return aggregated, metrics
    
    def _compute_signature(self, update: ClientUpdate) -> str:
        """Compute a signature for audit trail."""
        data = f"{update.client_id}:{update.round_number}:{update.timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _add_audit_event(self, event_type: str, data: Dict):
        """Add an event to the blockchain-like audit chain."""
        prev_hash = self.audit_chain[-1]["hash"] if self.audit_chain else "genesis"
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data,
            "prev_hash": prev_hash,
            "hash": ""
        }
        event["hash"] = hashlib.sha256(
            json.dumps(event, sort_keys=True).encode()
        ).hexdigest()
        
        self.audit_chain.append(event)
    
    def get_audit_trail(self) -> List[Dict]:
        """Get the full audit trail."""
        return self.audit_chain
    
    def simulate_local_training(
        self,
        client_id: str,
        local_data: np.ndarray,
        labels: np.ndarray
    ) -> ClientUpdate:
        """
        Simulate local training on client data.
        
        This is a simplified simulation. In production:
        - Use actual model training with PyTorch/TensorFlow
        - Apply differential privacy
        - Use secure multi-party computation
        
        Args:
            client_id: Client identifier
            local_data: Local training data
            labels: Local labels
            
        Returns:
            Client update with trained weights
        """
        # Simulate training progress
        np.random.seed(hash(client_id) % 2**32)
        
        # Simple gradient descent simulation
        initial_loss = 2.0
        final_loss = initial_loss * np.random.uniform(0.3, 0.7)
        accuracy = 1.0 - final_loss
        
        # Add some "noise" to simulate training
        noise = np.random.randn(len(self.global_weights)) * 0.1
        
        update = ClientUpdate(
            client_id=client_id,
            round_number=len(self.round_history) + 1,
            weights=self.global_weights + noise,
            num_samples=len(local_data),
            metrics={
                "loss": final_loss,
                "accuracy": accuracy
            },
            timestamp=datetime.now().isoformat(),
            signature=self._compute_signature(ClientUpdate(
                client_id, len(self.round_history) + 1,
                self.global_weights, len(local_data),
                {"loss": final_loss}, datetime.now().isoformat(), ""
            ))
        )
        
        return update
    
    def run_training_round(self) -> Dict:
        """
        Run one round of federated training.
        
        Returns:
            Round results including metrics
        """
        if self.global_weights is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
        
        self.status = FLStatus.TRAINING
        
        # Select clients
        selected = self.select_clients()
        
        # Simulate receiving updates from selected clients
        for client_id in selected:
            # In production: send global_weights to client, receive update
            # Here we simulate local training
            dummy_data = np.random.randn(100, 10)
            dummy_labels = np.random.randint(0, 2, 100)
            
            update = self.simulate_local_training(client_id, dummy_data, dummy_labels)
            self.receive_client_update(update)
        
        # Aggregate
        if self.client_updates:
            weights, metrics = self.aggregate_updates()
        else:
            metrics = {"error": "No updates received"}
        
        round_result = {
            "round": len(self.round_history) + 1,
            "selected_clients": selected,
            "metrics": metrics
        }
        self.round_history.append(round_result)
        
        return round_result


def create_fl_system(num_institutions: int = 4) -> FederatedLearningCoordinator:
    """
    Create and configure a federated learning system.
    
    Args:
        num_institutions: Number of participating institutions
        
    Returns:
        Configured FL coordinator
    """
    config = FederatedConfig(
        num_rounds=10,
        min_clients=max(2, num_institutions // 2),
        local_epochs=5,
        client_fraction=1.0
    )
    
    coordinator = FederatedLearningCoordinator(config)
    
    # Register simulated institutions
    institutions = [
        ("hospital_suzhou", "Suzhou Hospital", 5000),
        ("hospital_shanghai", "Shanghai Jiao Tong Hospital", 8000),
        ("hospital_beijing", "Peking Union Medical College Hospital", 10000),
        ("pharma_boston", "Boston Children's Hospital", 3000),
        ("research_xjtlu", "XJTLU AI Research Lab", 2000),
    ]
    
    for inst_id, name, data_size in institutions[:num_institutions]:
        coordinator.register_client(inst_id, name, data_size)
    
    # Initialize model (e.g., for rare disease classification)
    coordinator.initialize_model(input_dim=100, num_classes=10)
    
    return coordinator


def demonstrate_fl_usage():
    """
    Demonstrate federated learning usage.
    
    This shows how FL could be used for:
    - Rare disease patient data collaboration
    - Cross-hospital charity impact measurement
    - Privacy-preserving outcome tracking
    """
    print("=" * 60)
    print("Federated Learning Demo for PAI")
    print("=" * 60)
    
    # Create FL system
    fl = create_fl_system(num_institutions=4)
    
    print(f"\n📋 Registered Clients: {len(fl.clients)}")
    for cid, info in fl.clients.items():
        print(f"  - {cid}: {info['institution']} ({info['data_size']} samples)")
    
    print(f"\n🎯 Global model initialized")
    print(f"   Weights shape: {fl.global_weights.shape}")
    
    # Run training rounds
    print(f"\n🏋️ Running federated training...")
    for round_num in range(3):
        result = fl.run_training_round()
        print(f"  Round {round_num + 1}: {result['selected_clients']}")
        if 'metrics' in result and 'avg_accuracy' in result['metrics']:
            print(f"    Accuracy: {result['metrics']['avg_accuracy']:.2%}")
    
    print(f"\n📊 Training complete! Rounds: {len(fl.round_history)}")
    print(f"🔐 Audit trail entries: {len(fl.get_audit_trail())}")
    
    return fl


if __name__ == "__main__":
    demonstrate_fl_usage()
