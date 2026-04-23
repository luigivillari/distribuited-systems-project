"""
agents.py
=========
Fase 2 — Agent Modelling: ResourceAgent e TaskAgent come Ray Actors
"""

import ray
import time
import random
from typing import Dict, List, Optional, Tuple

from protocol import (
    A2AMessage, MessageType, TaskRequirements, ResourceOffer,
    PlacementPolicy, score_offer,
    make_propose, make_reject, make_accept, make_counter_offer, make_inform_done
)
from crdt_catalogue import ResourceCatalogue


# ═══════════════════════════════════════════════════════════
#  RESOURCE AGENT
# ═══════════════════════════════════════════════════════════

@ray.remote
class ResourceAgent:
    def __init__(self, node_id: str, total_cpu: float, total_memory_mb: float,
                 latency_ms: float, energy_score: float):

        self.node_id = node_id
        self.total_cpu = total_cpu
        self.total_memory_mb = total_memory_mb
        self.base_latency_ms = latency_ms
        self.energy_score = energy_score

        self.available_cpu = total_cpu
        self.available_memory_mb = total_memory_mb
        self.current_latency_ms = latency_ms

        self.active_tasks: Dict[str, Tuple[float, float, float]] = {}
        self.negotiation_log: List[dict] = []

        self.catalogue = ResourceCatalogue(owner_node_id=node_id)
        self.catalogue.upsert_node(
            node_id=node_id, cpu=self.available_cpu,
            memory_mb=self.available_memory_mb, latency_ms=self.current_latency_ms,
            energy_score=self.energy_score, active_tasks=0, is_online=True
        )
        self.peer_agents: List = []

        print(f"[{node_id}] ResourceAgent avviato: "
              f"CPU={total_cpu}, MEM={total_memory_mb}MB, "
              f"LAT={latency_ms}ms, ENERGY={energy_score:.2f}")

    def register_peers(self, peers: list):
        self.peer_agents = peers

    def get_catalogue_snapshot(self) -> dict:
        return self.catalogue.state_snapshot()

    def get_catalogue_object(self) -> ResourceCatalogue:
        return self.catalogue

    def sync_catalogue(self, remote_catalogue: ResourceCatalogue):
        before_clock = self.catalogue._lamport_clock
        self.catalogue.merge(remote_catalogue)
        after_clock = self.catalogue._lamport_clock
        if after_clock > before_clock:
            print(f"[{self.node_id}] CRDT sync: clock {before_clock} -> {after_clock}")

    def receive_cfp(self, cfp: A2AMessage) -> Optional[A2AMessage]:
        req = cfp.requirements
        start_time = time.time()

        jitter = random.uniform(-5, 15)
        self.current_latency_ms = max(1.0, self.base_latency_ms + jitter)

        can_satisfy_cpu = self.available_cpu >= req.cpu_cores
        can_satisfy_mem = self.available_memory_mb >= req.memory_mb
        can_satisfy_lat = self.current_latency_ms <= req.max_latency_ms

        log_entry = {
            "event": "cfp_received", "node_id": self.node_id,
            "task_id": cfp.task_id, "conv_id": cfp.conversation_id,
            "timestamp": start_time, "available_cpu": self.available_cpu,
            "available_mem": self.available_memory_mb,
            "required_cpu": req.cpu_cores, "required_mem": req.memory_mb,
        }

        if can_satisfy_cpu and can_satisfy_mem and can_satisfy_lat:
            offer = ResourceOffer(
                node_id=self.node_id, available_cpu=self.available_cpu,
                available_memory_mb=self.available_memory_mb,
                estimated_latency_ms=self.current_latency_ms,
                energy_cost_score=self.energy_score,
            )
            response = make_propose(self.node_id, cfp.sender_id,
                                    cfp.task_id, cfp.conversation_id, offer)
            log_entry["response"] = "PROPOSE"

        elif (self.available_cpu >= req.cpu_cores * 0.7 and
              self.available_memory_mb >= req.memory_mb * 0.7):
            offer = ResourceOffer(
                node_id=self.node_id, available_cpu=self.available_cpu,
                available_memory_mb=self.available_memory_mb,
                estimated_latency_ms=self.current_latency_ms,
                energy_cost_score=self.energy_score,
            )
            response = make_counter_offer(self.node_id, cfp.sender_id,
                                          cfp.task_id, cfp.conversation_id, offer)
            log_entry["response"] = "COUNTER_OFFER"

        else:
            reasons = []
            if not can_satisfy_cpu:
                reasons.append(f"CPU: {self.available_cpu:.1f}/{req.cpu_cores} richiesti")
            if not can_satisfy_mem:
                reasons.append(f"MEM: {self.available_memory_mb:.0f}/{req.memory_mb}MB richiesti")
            if not can_satisfy_lat:
                reasons.append(f"LAT: {self.current_latency_ms:.0f}ms > {req.max_latency_ms}ms max")
            response = make_reject(self.node_id, cfp.sender_id,
                                   cfp.task_id, cfp.conversation_id,
                                   " | ".join(reasons))
            log_entry["response"] = "REJECT"
            log_entry["reason"] = " | ".join(reasons)

        self.negotiation_log.append(log_entry)
        return response

    def receive_reject(self, reject_msg: A2AMessage):
        self.negotiation_log.append({
            "event": "proposal_rejected", "node_id": self.node_id,
            "task_id": reject_msg.task_id, "reason": reject_msg.reason,
            "timestamp": time.time(),
        })

    def receive_accept(self, accept_msg: A2AMessage) -> A2AMessage:
        task_id = accept_msg.task_id
        cpu_alloc = min(2.0, self.available_cpu)
        mem_alloc = min(512.0, self.available_memory_mb)

        self.available_cpu -= cpu_alloc
        self.available_memory_mb -= mem_alloc
        self.active_tasks[task_id] = (cpu_alloc, mem_alloc, time.time())

        self.catalogue.upsert_node(
            node_id=self.node_id, cpu=self.available_cpu,
            memory_mb=self.available_memory_mb, latency_ms=self.current_latency_ms,
            energy_score=self.energy_score, active_tasks=len(self.active_tasks),
            is_online=True,
        )

        print(f"[{self.node_id}] ACCEPT ricevuto per task={task_id}. "
              f"CPU rimasta: {self.available_cpu:.1f}, MEM: {self.available_memory_mb:.0f}MB")

        self.negotiation_log.append({
            "event": "task_accepted", "node_id": self.node_id,
            "task_id": task_id, "cpu_allocated": cpu_alloc,
            "mem_allocated": mem_alloc, "timestamp": time.time(),
        })

        return make_inform_done(self.node_id, accept_msg.sender_id,
                                task_id, accept_msg.conversation_id)

    def mark_offline_self(self):
        """Marca questo nodo come offline nel proprio catalogo CRDT."""
        self.catalogue.mark_offline(self.node_id)
        print(f"[{self.node_id}] Marcato OFFLINE nel catalogo CRDT")

    def complete_task(self, task_id: str):
        if task_id in self.active_tasks:
            cpu_alloc, mem_alloc, start_t = self.active_tasks.pop(task_id)
            self.available_cpu = min(self.total_cpu, self.available_cpu + cpu_alloc)
            self.available_memory_mb = min(self.total_memory_mb,
                                           self.available_memory_mb + mem_alloc)
            self.catalogue.upsert_node(
                node_id=self.node_id, cpu=self.available_cpu,
                memory_mb=self.available_memory_mb, latency_ms=self.current_latency_ms,
                energy_score=self.energy_score, active_tasks=len(self.active_tasks),
                is_online=True,
            )
            print(f"[{self.node_id}] Task {task_id} completato in "
                  f"{time.time()-start_t:.2f}s")

    def get_state(self) -> dict:
        return {
            "node_id": self.node_id,
            "available_cpu": self.available_cpu,
            "available_memory_mb": self.available_memory_mb,
            "current_latency_ms": self.current_latency_ms,
            "energy_score": self.energy_score,
            "active_tasks": len(self.active_tasks),
            "total_negotiations": len(self.negotiation_log),
        }

    def get_negotiation_log(self) -> List[dict]:
        return self.negotiation_log


# ═══════════════════════════════════════════════════════════
#  TASK AGENT
# ═══════════════════════════════════════════════════════════

@ray.remote
class TaskAgent:
    def __init__(self, task_id: str, requirements: TaskRequirements,
                 policy: PlacementPolicy = PlacementPolicy.BALANCED):

        self.task_id = task_id
        self.requirements = requirements
        self.policy = policy
        self.status = "pending"
        self.placed_on: Optional[str] = None
        self.placement_latency_ms: Optional[float] = None
        self.received_proposals: Dict[str, A2AMessage] = {}
        self.rejected_by: List[str] = []
        self.event_log: List[dict] = []

        print(f"[TaskAgent:{task_id}] Creato. "
              f"Policy={policy.value}, CPU={requirements.cpu_cores}, "
              f"MEM={requirements.memory_mb}MB, LAT_MAX={requirements.max_latency_ms}ms")

    def place(self, resource_agents: list) -> dict:
        from protocol import make_cfp
        t_start = time.time()
        self.status = "negotiating"

        cfp = make_cfp(f"task-{self.task_id}", self.task_id, self.requirements)
        self.event_log.append({"event": "cfp_sent", "timestamp": t_start,
                                "n_agents": len(resource_agents)})
        print(f"[TaskAgent:{self.task_id}] -> CFP inviata a {len(resource_agents)} nodi")

        response_refs = [agent.receive_cfp.remote(cfp) for agent in resource_agents]
        responses = ray.get(response_refs)

        proposals = []
        for resp in responses:
            if resp is None:
                continue
            if resp.msg_type in (MessageType.PROPOSE, MessageType.COUNTER_OFFER):
                score = score_offer(resp.offer, self.policy)
                resp.offer.score = score
                proposals.append(resp)
                self.received_proposals[resp.sender_id] = resp
                print(f"[TaskAgent:{self.task_id}] <- {resp.msg_type.value} da "
                      f"{resp.sender_id}: score={score:.3f}, "
                      f"lat={resp.offer.estimated_latency_ms:.0f}ms")
            elif resp.msg_type == MessageType.REJECT:
                self.rejected_by.append(resp.sender_id)
                print(f"[TaskAgent:{self.task_id}] <- REJECT da "
                      f"{resp.sender_id}: {resp.reason}")

        if not proposals:
            self.status = "failed"
            t_end = time.time()
            result = {
                "task_id": self.task_id, "status": "failed",
                "reason": "no_proposals",
                "placement_latency_ms": (t_end - t_start) * 1000,
                "proposals_received": 0,
            }
            print(f"[TaskAgent:{self.task_id}] x Nessuna proposta ricevuta.")
            return result

        best   = max(proposals, key=lambda r: r.offer.score)
        losers = [r for r in proposals if r.sender_id != best.sender_id]

        print(f"[TaskAgent:{self.task_id}] -> Vincitore: {best.sender_id} "
              f"(score={best.offer.score:.3f})")

        winner_idx = next(i for i, a in enumerate(resource_agents)
                          if ray.get(a.get_state.remote())["node_id"] == best.sender_id)
        accept_msg = make_accept(f"task-{self.task_id}", best.sender_id,
                                 self.task_id, cfp.conversation_id)
        ray.get(resource_agents[winner_idx].receive_accept.remote(accept_msg))

        for loser_resp in losers:
            loser_idx = next(i for i, a in enumerate(resource_agents)
                             if ray.get(a.get_state.remote())["node_id"] == loser_resp.sender_id)
            reject_msg = make_reject(f"task-{self.task_id}", loser_resp.sender_id,
                                     self.task_id, cfp.conversation_id,
                                     "better_offer_available")
            resource_agents[loser_idx].receive_reject.remote(reject_msg)

        t_end = time.time()
        self.status = "placed"
        self.placed_on = best.sender_id
        self.placement_latency_ms = (t_end - t_start) * 1000

        result = {
            "task_id": self.task_id, "status": "placed",
            "placed_on": self.placed_on, "policy": self.policy.value,
            "score": best.offer.score,
            "estimated_latency_ms": best.offer.estimated_latency_ms,
            "energy_score": best.offer.energy_cost_score,
            "placement_latency_ms": self.placement_latency_ms,
            "proposals_received": len(proposals),
            "rejections_received": len(self.rejected_by),
        }
        self.event_log.append({"event": "placement_success", **result})
        print(f"[TaskAgent:{self.task_id}] Piazzato su {self.placed_on} "
              f"in {self.placement_latency_ms:.1f}ms")
        return result

    def get_status(self) -> dict:
        return {
            "task_id": self.task_id, "status": self.status,
            "placed_on": self.placed_on,
            "placement_latency_ms": self.placement_latency_ms,
            "policy": self.policy.value,
        }