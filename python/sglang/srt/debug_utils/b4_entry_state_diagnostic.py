"""Bounded, read-only B4 entry-state comparison for two declared requests.

Only the matching source-patch fixture enables this observer. Byte differences
are observations, not a recovery failure or a numerical-cause attribution.
No diagnostic value is fed back to the native model or its cache.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

ANCHORS = (0, 1, 3, 7)
LENGTHS = (149, 149, 151, 155)
GUARD_BYTES = 512 << 20
PRIVATE_BYTES = 274661916
ENTRY_METADATA_LIMIT = 16 << 10
SELECTED = (107, 187)
PAYLOAD_ANCHORS = (1, 3, 7)
PAYLOAD_EXTRA_BYTES = 202690560
PAYLOAD_LOGICAL_BYTES = 404971520
PAYLOAD_STAGING_BYTES = 2097152
PAYLOAD_COMPARE_BYTES = 64 << 10
PAYLOAD_FILE_OVERHEAD = 64 << 10
PAYLOAD_MANIFEST_LIMIT = 1 << 20


def file_sha256(path):
    digest = hashlib.sha256()
    workspace = bytearray(64 << 10)
    with path.open("rb") as stream:
        while (size := stream.readinto(workspace)):
            digest.update(memoryview(workspace)[:size])
    return digest.hexdigest()


def payload_contract():
    return dict(client=1, indices=list(SELECTED), anchors=list(PAYLOAD_ANCHORS),
                additional_private_bytes=PAYLOAD_EXTRA_BYTES,
                logical_tensor_bytes=PAYLOAD_LOGICAL_BYTES,
                cpu_tensor_staging_limit_bytes=2 * PAYLOAD_STAGING_BYTES + PAYLOAD_COMPARE_BYTES + 8,
                individual_tensor_staging_limit_bytes=PAYLOAD_STAGING_BYTES,
                comparison_workspace_limit_bytes=PAYLOAD_COMPARE_BYTES,
                per_file_overhead_limit_bytes=PAYLOAD_FILE_OVERHEAD,
                manifest_limit_bytes=PAYLOAD_MANIFEST_LIMIT,
                export="post2_fully_idle_only")


def byte_mismatch_count(actual, reference):
    require(actual.device.type == reference.device.type == "cpu" and actual.dtype == reference.dtype
            and actual.shape == reference.shape and actual.is_contiguous() and reference.is_contiguous(),
            "payload comparison operand identity")
    left, right = actual.view(torch.uint8).flatten(), reference.view(torch.uint8).flatten()
    workspace = torch.empty(min(left.numel(), PAYLOAD_COMPARE_BYTES), dtype=torch.bool)
    differences = 0
    for start in range(0, left.numel(), PAYLOAD_COMPARE_BYTES):
        end = min(start + PAYLOAD_COMPARE_BYTES, left.numel())
        torch.ne(left[start:end], right[start:end], out=workspace[:end - start])
        differences += int(torch.count_nonzero(workspace[:end - start]))
    return differences


def require(condition, reason):
    if not condition:
        raise RuntimeError("B4 entry-state diagnostic: " + reason)


@dataclass(frozen=True)
class Layout:
    layers: int = 30
    temporal: tuple = (32, 128, 128)
    conv: tuple = (8192, 3)
    kv_layers: int = 10
    kv_shape: tuple = (2, 256)
    max_length: int = 155

    def specs(self):
        state_bytes = math.prod(self.temporal) * 4
        return {
            "temporal": ((4, self.layers, *self.temporal), torch.float32),
            "conv": ((4, self.layers, *self.conv), torch.bfloat16),
            "kv": ((4, self.kv_layers, 2, self.max_length, *self.kv_shape), torch.bfloat16),
            "differences": ((4, 2 * self.layers + 2 * self.kv_layers), torch.int64),
            "temporal_work": ((1, *self.temporal), torch.float32),
            "conv_work": ((1, *self.conv), torch.bfloat16),
            "kv_work": ((self.max_length, *self.kv_shape), torch.bfloat16),
            "compare_work": ((state_bytes,), torch.bool),
            "rank": ((self.max_length,), torch.int64),
            "offset_work": ((self.max_length,), torch.int64),
            "index_work": ((self.max_length,), torch.int32),
        }

    def nbytes(self):
        return sum(math.prod(shape) * dtype.itemsize for shape, dtype in self.specs().values())

    def comparison_specs(self):
        return {"actual_" + name: ((3, *shape[1:]), dtype)
                for name, (shape, dtype) in self.specs().items()
                if name in ("temporal", "conv", "kv")}

    def comparison_nbytes(self):
        return sum(math.prod(shape) * dtype.itemsize for shape, dtype in self.comparison_specs().values())


class EntryBuffers:
    """Immutable reference banks; scratch is consumed on one native stream."""

    def __init__(self, device, layout=Layout(), retain_comparisons=False):
        self.layout = layout
        self.retain_comparisons = retain_comparisons
        specs = layout.specs()
        if retain_comparisons:
            specs.update(layout.comparison_specs())
        self.buffers = {k: torch.empty(shape, dtype=dtype, device=device)
                        for k, (shape, dtype) in specs.items()}
        self.buffers["rank"].copy_(torch.arange(layout.max_length, device=device))
        self.references = set()
        self.comparisons = set()

    def ordered_indices(self, values, offset, length):
        require(0 < length <= self.layout.max_length, "ordered-index bound")
        work = self.buffers["offset_work"][:length]
        torch.add(self.buffers["rank"][:length], offset, out=work)
        target = self.buffers["index_work"][:length]
        torch.index_select(values, 0, work, out=target)
        return target

    def _count(self, actual, expected, destination):
        require(actual.shape == expected.shape and actual.dtype == expected.dtype,
                "byte-comparison shape/dtype")
        a, b = actual.contiguous().view(torch.uint8).flatten(), expected.contiguous().view(torch.uint8).flatten()
        scratch = self.buffers["compare_work"][:a.numel()]
        require(scratch.numel() == a.numel(), "comparison workspace bound")
        torch.ne(a, b, out=scratch)
        torch.sum(scratch, dim=0, dtype=torch.int64, out=destination)

    def capture(self, anchor, reference, state, state_index, kv_pool, full_layers, indices):
        require(anchor in range(4), "anchor index")
        require((anchor not in self.references) if reference else (anchor in self.references),
                "duplicate/missing immutable reference")
        retained = self.retain_comparisons and not reference and anchor > 0
        require(not retained or anchor not in self.comparisons, "comparison bank overwrite")
        s, buf = self.layout, self.buffers
        n = indices.numel()
        require(0 < n <= s.max_length, "KV prefix bound")
        require(len(full_layers) == s.kv_layers, "full-attention layer count")
        require(tuple(state.temporal.shape[2:]) == s.temporal
                and state.temporal.shape[0] == s.layers
                and tuple(state.conv[0].shape[2:]) == s.conv,
                "native recurrent layout changed")
        require(state.temporal.dtype == torch.float32 and state.conv[0].dtype == torch.bfloat16,
                "native recurrent precision changed")
        # Invalid indices are retained in metadata and rejected at the existing
        # result fence. Clamp diagnostic reads only, never native indices.
        safe_state = state_index.clamp(0, state.temporal.shape[1] - 1)
        counts = buf["differences"][anchor]
        counts.fill_(-1)
        for i in range(s.layers):
            for name, source, column in (
                ("temporal", state.temporal[i], i),
                ("conv", state.conv[0][i], s.layers + i),
            ):
                expected = buf[name][anchor, i:i + 1]
                out = expected if reference else buf[name + "_work"]
                if retained:
                    out = buf["actual_" + name][anchor - 1, i:i + 1]
                torch.index_select(source, 0, safe_state, out=out)
                if not reference:
                    self._count(out, expected, counts[column])
        for i, layer in enumerate(full_layers):
            for component, source in enumerate(kv_pool.get_kv_buffer(layer)):
                require(tuple(source.shape[1:]) == s.kv_shape and source.dtype == torch.bfloat16,
                        "native KV layout/precision changed")
                expected = buf["kv"][anchor, i, component, :n]
                out = expected if reference else buf["kv_work"][:n]
                if retained:
                    out = buf["actual_kv"][anchor - 1, i, component, :n]
                safe_kv = indices.clamp(0, source.shape[0] - 1)
                torch.index_select(source, 0, safe_kv, out=out)
                if not reference:
                    self._count(out, expected, counts[2 * s.layers + 2 * i + component])
        if reference:
            self.references.add(anchor)
        elif retained:
            self.comparisons.add(anchor)
        return counts.clone()  # Result owns these bytes before scratch is reused.


class EntryObserver:
    def __init__(self, runner, observer):
        require(Layout().nbytes() == PRIVATE_BYTES, "declared private byte budget")
        requested = observer.config.get("entry_state_payload")
        self.payload_enabled = requested is not None
        require(not self.payload_enabled or requested == payload_contract(), "exact payload declaration")
        require(Layout().comparison_nbytes() == PAYLOAD_EXTRA_BYTES, "payload bank byte declaration")
        self.private_bytes = PRIVATE_BYTES + (PAYLOAD_EXTRA_BYTES if self.payload_enabled else 0)
        require(not self.payload_enabled or observer.config.get("entry_state") == dict(
            client=1, indices=list(SELECTED), anchors=list(ANCHORS), private_tensor_bytes=self.private_bytes,
            incremental_guard_bytes=GUARD_BYTES), "exact four-anchor payload storage declaration")
        self.payload_events = {}
        self.payload_status = dict(enabled=self.payload_enabled, status="not_started")
        self.runner, self.observer = runner, observer
        pool = runner.req_to_token_pool
        state = pool.get_speculative_mamba2_params_all_layers()
        expected_layers = tuple(i for i in range(40) if i % 4 != 3)
        self.full_layers = tuple(range(3, 40, 4))
        require(pool.size == 32 and pool.mamba_map == dict(zip(expected_layers, range(30)))
                and tuple(state.temporal.shape) == (30, 33, 32, 128, 128)
                and tuple(state.conv[0].shape) == (30, 33, 8192, 3)
                and state.intermediate_ssm is None and state.compact_ssm_k is not None,
                "frozen native B4 state pool")
        self.pool, self.state = pool, state
        self.kv_pool = runner.attn_backend.token_to_kv_pool
        require(tuple(runner.attn_backend.full_attn_layers) == self.full_layers,
                "frozen full-attention mapping")
        require(self.kv_pool.page_size == 1, "page size changed")
        free, _ = torch.cuda.mem_get_info(runner.device)
        require(free >= self.private_bytes + GUARD_BYTES + (4 << 20), "preallocation memory guard")
        before = torch.cuda.memory_allocated(runner.device)
        self.storage = EntryBuffers(runner.device, retain_comparisons=self.payload_enabled)
        self.allocator_increment = torch.cuda.memory_allocated(runner.device) - before
        require(torch.cuda.mem_get_info(runner.device)[0] >= GUARD_BYTES + (4 << 20),
                "postallocation memory guard")
        self.requests, self.enqueued, self.published, self.records = {}, set(), set(), {}
        self.stream = None

    def _request(self, batch):
        selected = []
        for row, req in enumerate(batch.reqs):
            known = self.observer.rows.get(req.rid)
            if known and known["repetition"] == 1 and known["phase"] == "measure" and known["idx"] in SELECTED:
                require(len(known["input_ids"]) == 149, "selected prompt length")
                require(not req.is_retracted, "selected request retracted")
                self.observer.request_row(req)
                idx = known["idx"]
                old = self.requests.setdefault(idx, req)
                require(old is req, "request generation replaced")
                require(all(other is not req for key, other in self.requests.items() if key != idx),
                        "two catalog generations share request object")
                selected.append((row, req, idx))
        require(len(selected) <= 1, "two watched requests simultaneously active")
        return selected[0] if selected else None

    def before_verify(self, batch, forward, can_graph):
        selection = self._request(batch)
        if selection is None:
            return
        row, req, idx = selection
        host = forward.seq_lens_cpu
        require(isinstance(host, torch.Tensor) and host.device.type == "cpu", "native CPU sequence metadata")
        length = int(host[row])
        if length not in LENGTHS[1:]:
            return
        anchor = LENGTHS.index(length, 1)
        reason = "matched entry requires native batch/graph32"
        if not (can_graph and len(batch.reqs) == 32):
            # Failure-only host metadata: no device read, event, or snapshot ID.
            context = dict(rid=req.rid, idx=idx, anchor=ANCHORS[anchor],
                           native_seq_len=length, selected_row=row,
                           native_batch_size=len(batch.reqs),
                           forward_batch_size=forward.batch_size, can_graph=can_graph,
                           forward_metadata_ready=forward.forward_metadata_ready,
                           req_pool_index=req.req_pool_idx,
                           last_snapshot=self.observer.serial,
                           pending_count=len(self.observer.pending))
            reason += "; context=" + json.dumps(context, sort_keys=True)
        require(can_graph and len(batch.reqs) == 32, reason)
        require(forward.forward_metadata_ready, "entry before native graph plan")
        require(not hasattr(forward, "_b4_entry_selection"), "duplicate entry selection")
        forward._b4_entry_selection = (row, req, idx, anchor)

    def _stream_guard(self):
        require(not torch.cuda.is_current_stream_capturing(), "entry observer cannot enter capture")
        stream = torch.cuda.current_stream(self.runner.device).cuda_stream
        if self.stream is None:
            self.stream = stream
        require(stream == self.stream, "native producing stream changed")

    def _capture(self, selection, forward, graph=None):
        row, req, idx, anchor = selection
        key = (idx, anchor)
        require(key not in self.enqueued, "duplicate entry anchor")
        self._stream_guard()
        n, length = forward.batch_size, LENGTHS[anchor]
        hybrid = self.runner.attn_backend if graph is None else graph.attn_backend
        linear, full = hybrid.linear_attn_backend, hybrid.full_attn_backend
        metadata = linear.forward_metadata
        source = forward if graph is None else graph.buffers
        require(graph is None or (graph.bs == 32 and graph.raw_bs == 32
                and graph.model_runner is self.runner
                and metadata.mamba_cache_indices.data_ptr() == linear.state_indices_list[31].data_ptr()),
                "executed graph/state-index identity")
        indices = metadata.mamba_cache_indices[:n]
        tensors = {
            "entry_req_indices": source.req_pool_indices[:n],
            "entry_resolved_state_indices": indices,
            "entry_mapping_state_indices": self.pool.get_mamba_indices(source.req_pool_indices[:n]),
            "entry_query_bounds": metadata.query_start_loc[:n + 1],
            "entry_seq_lens": source.seq_lens[:n],
            "entry_positions": source.positions[:n * 5 if graph is not None else length],
            "entry_input_ids": source.input_ids[:n * 5 if graph is not None else length],
            "entry_request_table_indices": self.pool.req_to_token[req.req_pool_idx, :length],
        }
        mrope = getattr(source, "mrope_positions", None)
        if mrope is not None:
            require(mrope.ndim == 2 and mrope.shape[0] == 3, "native mRoPE layout")
            tensors["entry_mrope_positions"] = mrope[..., :n * 5 if graph is not None else length]
        if graph is not None:
            # Replay replans these graph-owned buffers without rebinding the
            # eager forward_metadata left by the preceding native prefill.
            wrappers = full.prefill_cuda_graph_metadata[graph.bs]
            require(len(wrappers) == 1, "executed graph KV wrapper identity")
            wrapper = wrappers[0]
            require(wrapper._custom_mask_buf is not None and wrapper._mask_indptr_buf is not None,
                    "native verify packed mask absent")
            tensors.update(
                entry_intermediate_state_indices=linear.verify_intermediate_state_indices[:n],
                entry_kv_indptr=wrapper._paged_kv_indptr_buf[:n + 1],
                entry_qo_indptr=wrapper._qo_indptr_buf[:n + 1],
                entry_kv_last_page_len=wrapper._paged_kv_last_page_len_buf[:n],
                entry_mask_indptr=wrapper._mask_indptr_buf[:n + 1],
            )
            kv_indices = self.storage.ordered_indices(wrapper._paged_kv_indices_buf,
                                                      wrapper._paged_kv_indptr_buf[row], length).clone()
            mask_len = (5 * (length + 5) + 7) // 8
            offsets = self.storage.buffers["offset_work"][:mask_len]
            torch.add(self.storage.buffers["rank"][:mask_len], wrapper._mask_indptr_buf[row], out=offsets)
            tensors["entry_packed_mask"] = torch.index_select(wrapper._custom_mask_buf, 0, offsets)
        else:
            kv_indices = tensors["entry_request_table_indices"].to(torch.int32)
        tensors["entry_ordered_kv_indices"] = kv_indices
        current_state = self.pool.get_speculative_mamba2_params_all_layers()
        tensors["entry_byte_differences"] = self.storage.capture(
            anchor, idx == 107, current_state, indices[row:row + 1], self.kv_pool, self.full_layers, kv_indices)
        require(sum(t.numel() * t.element_size() for t in tensors.values()) <= ENTRY_METADATA_LIMIT,
                "entry metadata byte bound")
        # Own all dynamic metadata before graph replay can reuse its inputs.
        tensors = {name: value.detach().clone() for name, value in tensors.items()}
        meta = dict(rid=req.rid, idx=idx, anchor=ANCHORS[anchor], prefix_length=length,
                    row=row, generation=SELECTED.index(idx) + 1,
                    reference=idx == 107, req_pool_index=req.req_pool_idx,
                    kv_source="executed_graph_wrapper_prefix" if graph is not None else "request_table_post_prefill",
                    graph_bucket=32 if graph is not None else None,
                    kv_capacity=self.kv_pool.get_kv_buffer(self.full_layers[0])[0].shape[0],
                    linear_layers=list(self.pool.mamba_map), full_attention_layers=list(self.full_layers),
                    tensor_comparison="exact mismatched byte counts; -1 means reference captured, not compared")
        self.enqueued.add(key)
        return tensors, meta

    def graph_entry(self, graph, forward):
        selection = getattr(forward, "_b4_entry_selection", None)
        if selection is not None:
            require(not hasattr(forward, "_b4_entry_capture"), "duplicate graph entry")
            forward._b4_entry_capture = self._capture(selection, forward, graph)

    def take_verify(self, forward):
        selection = getattr(forward, "_b4_entry_selection", None)
        if selection is None:
            return {}, None
        require(hasattr(forward, "_b4_entry_capture"), "selected graph entry was not observed")
        capture = forward._b4_entry_capture
        del forward._b4_entry_capture
        del forward._b4_entry_selection
        return capture

    def post_prefill(self, batch, forward):
        selection = self._request(batch)
        if selection is None:
            return {}, None
        row, req, idx = selection
        require(len(batch.reqs) == 1 and forward.extend_prefix_lens_cpu == [0]
                and forward.extend_seq_lens_cpu == [149], "matched solo prefill shape")
        return self._capture((row, req, idx, 0), forward)

    def publish(self, req, record, completion=None):
        meta = record["metadata"].get("entry_state")
        if meta is None or meta["rid"] != req.rid:
            return
        data, row = record["tensors"], meta["row"]
        key = (meta["idx"], ANCHORS.index(meta["anchor"]))
        problems = record["problems"]
        if key not in self.enqueued or key in self.published:
            problems.append("entry anchor not uniquely owned")
        if record["start_position"] != meta["anchor"]:
            problems.append("entry sequence does not bind native publication window")
        owners = data["entry_resolved_state_indices"]
        if (owners != data["entry_mapping_state_indices"] or len(owners) != len(set(owners))
                or any(not 1 <= x <= 32 for x in owners)
                or owners[row] != data[f"request_mamba_pool_index_{row}"]):
            problems.append("entry recurrent owner mapping mismatch")
        if data["entry_req_indices"] != record["metadata"]["req_pool_indices_cpu"]:
            problems.append("entry executed request indices differ from owners")
        if data["entry_seq_lens"][row] != meta["prefix_length"]:
            problems.append("entry native CPU/device sequence disagreement")
        kv = data["entry_ordered_kv_indices"]
        if (kv != data["entry_request_table_indices"] or len(kv) != len(set(kv))
                or any(not 1 <= x < meta["kv_capacity"] for x in kv)):
            problems.append("entry ordered KV identity mismatch")
        length = meta["prefix_length"]
        if meta["anchor"]:
            bounds = data["entry_query_bounds"]
            if bounds != list(range(0, 161, 5)) or data["entry_qo_indptr"] != bounds:
                problems.append("entry query bounds mismatch")
            if data["entry_kv_indptr"][row + 1] - data["entry_kv_indptr"][row] != length + 5:
                problems.append("entry KV bounds mismatch")
            if data["entry_kv_last_page_len"][row] != 1:
                problems.append("entry page1 tail mismatch")
            if data["entry_intermediate_state_indices"] != list(range(32)):
                problems.append("entry compact intermediate ownership mismatch")
            mask_len = (5 * (length + 5) + 7) // 8
            if (data["entry_mask_indptr"][row + 1] - data["entry_mask_indptr"][row] != mask_len
                    or len(data["entry_packed_mask"]) != mask_len):
                problems.append("entry packed-mask bounds mismatch")
            if data["entry_positions"][row * 5:(row + 1) * 5] != list(range(length, length + 5)):
                problems.append("entry actual graph positions mismatch")
            if data["entry_input_ids"] != data["input_ids"]:
                problems.append("entry actual graph inputs differ from native verification")
        elif data["entry_query_bounds"] != [0, length] or data["entry_positions"] != list(range(length)):
            problems.append("entry solo-prefill positions/query bounds mismatch")
        counts = data["entry_byte_differences"]
        layout = self.storage.layout
        maxima = ([math.prod(layout.temporal) * 4] * layout.layers
                  + [math.prod(layout.conv) * 2] * layout.layers
                  + [length * math.prod(layout.kv_shape) * 2] * (2 * layout.kv_layers))
        if len(counts) != len(maxima) or any(
                value != -1 if meta["reference"] else not 0 <= value <= bound
                for value, bound in zip(counts, maxima)):
            problems.append("entry byte-comparison evidence bound")
        summary = dict(meta, snapshot=record["snapshot"], accepted_prefix_sha256=record["accepted_prefix_sha256"],
                       byte_differences=data["entry_byte_differences"],
                       selected_input=(data["entry_input_ids"][row * 5:(row + 1) * 5] if meta["anchor"] else data["entry_input_ids"]),
                       positions=(data["entry_positions"][row * 5:(row + 1) * 5] if meta["anchor"] else data["entry_positions"]),
                       mrope_positions=([axis[row * 5:(row + 1) * 5] if meta["anchor"] else axis
                                         for axis in data["entry_mrope_positions"]]
                                        if "entry_mrope_positions" in data else None),
                       # Only meaningful mask bits participate in input identity;
                       # the last packed byte may contain unused padding bits.
                       mask_bits=([bool(byte & (1 << bit)) for byte in data["entry_packed_mask"]
                                   for bit in range(8)][:5 * (length + 5)] if meta["anchor"] else None),
                       native_argmax=data[f"argmax_{row}"], top_ids=data[f"top_ids_{row}"], top_values=data[f"top_values_{row}"])
        if meta["idx"] == 187:
            ref = self.records.get((107, key[1]))
            if ref is None or any(summary[name] != ref[name] for name in (
                    "accepted_prefix_sha256", "selected_input", "positions", "mrope_positions", "mask_bits")):
                problems.append("entry comparison lacks matched selected inputs/prefix/mask")
        self.records[key] = summary
        self.published.add(key)
        record["entry_state"] = summary
        if getattr(self, "payload_enabled", False) and meta["anchor"] in PAYLOAD_ANCHORS:
            require(not problems and completion is not None and completion.query(),
                    "payload requires validated published completion event")
            require(key not in self.payload_events, "payload event generation replaced")
            self.payload_events[key] = completion
        record["limitations"] += " Entry-state byte comparison covers only the declared selected anchors; it is not an independent recurrent recovery oracle or root-cause proof."

    def export_payloads(self, eligible):
        """Only the existing fully-idle post-workload endpoint may request this."""
        if not getattr(self, "payload_enabled", False):
            return
        if self.payload_status["status"] == "failed":
            raise RuntimeError("B4 entry-state diagnostic: previous payload export failed; preserve partial files")
        if not eligible or self.payload_status["status"] == "complete":
            return
        expected = {(idx, a) for idx in SELECTED for a in range(4)}
        payload_keys = {(idx, a) for idx in SELECTED for a in (1, 2, 3)}
        require(self.published == self.enqueued == expected and set(self.records) == expected,
                "payload export requires complete entry coverage")
        require(self.storage.references == set(range(4)) and self.storage.comparisons == {1, 2, 3}
                and set(self.payload_events) == payload_keys
                and all(event.query() for event in self.payload_events.values()), "payload bank/event ownership")
        require(not torch.cuda.is_current_stream_capturing(), "payload export during capture")
        require(torch.cuda.mem_get_info(self.runner.device)[0] >= GUARD_BYTES + (4 << 20),
                "payload export physical headroom")
        directory = self.observer.directory / "entry-state-payloads"
        # Existing paths are never overwritten, including an earlier failed export.
        directory.mkdir(exist_ok=False)
        self.payload_status = dict(enabled=True, status="writing", directory=directory.name)
        manifest = dict(status="writing", performance_eligible=False, runtime=self.observer.runtime,
                        contract=payload_contract(), entries=[], files=[], logical_tensor_bytes=0)
        reference_files = {}
        try:
            runtime_root = Path(__file__).resolve().parents[4]
            manifest["source_sha256"] = {name: hashlib.sha256((runtime_root / name).read_bytes()).hexdigest()
                for name in ("python/sglang/srt/debug_utils/b4_entry_state_diagnostic.py",
                             "python/sglang/srt/debug_utils/b4_verify_diagnostic.py", "test/manual/b4_verify_diagnostic.yaml")}
            manifest["catalog_sha256"] = hashlib.sha256(json.dumps(self.observer.config, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            for idx, a in sorted(payload_keys):
                record = self.records[idx, a]
                record_path = self.observer.directory / f'{idx:04d}-{record["snapshot"]:06d}-verify.json'
                saved = json.loads(record_path.read_text())
                require(saved["runtime"] == self.observer.runtime and saved["problems"] == []
                        and saved["entry_state"] == record, "payload publication record changed")
                manifest["entries"].append(dict(record=record,
                    capture=record_path.name, capture_sha256=hashlib.sha256(record_path.read_bytes()).hexdigest(),
                    state_owner=saved["tensors"]["entry_resolved_state_indices"][record["row"]],
                    ordered_kv_indices=saved["tensors"]["entry_ordered_kv_indices"]))
                bank, bank_anchor = ("", a) if idx == 107 else ("actual_", a - 1)
                for name, layers in (("temporal", list(self.pool.mamba_map)), ("conv", list(self.pool.mamba_map)),
                                     ("kv", self.full_layers)):
                    for slot, layer in enumerate(layers):
                        for component in (range(2) if name == "kv" else (None,)):
                            value = self.storage.buffers[bank + name][bank_anchor, slot]
                            if component is not None:
                                value = value[component, :record["prefix_length"]]
                            layout = self.storage.layout
                            expected_shape = (layout.temporal if name == "temporal" else layout.conv
                                if name == "conv" else (record["prefix_length"], *layout.kv_shape))
                            require(tuple(value.shape) == expected_shape and value.dtype ==
                                    (torch.float32 if name == "temporal" else torch.bfloat16),
                                    "payload immutable bank shape/dtype")
                            nbytes = value.numel() * value.element_size()
                            require(nbytes <= PAYLOAD_STAGING_BYTES, "payload CPU staging bound")
                            # At most two compact CPU tensors plus the declared byte-compare workspace.
                            cpu = value.detach().to(device="cpu", copy=True).contiguous()
                            tensor_hash = hashlib.sha256(cpu.view(torch.uint8).numpy()).hexdigest()
                            hash_key = (a, name, layer, component)
                            if idx == 187:
                                column = (slot if name == "temporal" else layout.layers + slot
                                    if name == "conv" else 2 * layout.layers + 2 * slot + component)
                                reference_path, reference_file_hash = reference_files[hash_key]
                                require(file_sha256(reference_path) == reference_file_hash,
                                        "payload reference file changed")
                                reference = torch.load(reference_path, weights_only=True, map_location="cpu")
                                require(byte_mismatch_count(cpu, reference) == record["byte_differences"][column],
                                        "payload bytes contradict captured comparison")
                                del reference
                            filename = f'{idx:04d}-anchor{ANCHORS[a]}-{name}-layer{layer:02d}'
                            filename += (f'-{component}' if component is not None else '') + '.pt'
                            path = directory / filename
                            with path.open("xb") as stream:
                                torch.save(cpu, stream)  # Fail closed; generic dumper swallows save errors.
                            require(path.stat().st_size <= nbytes + PAYLOAD_FILE_OVERHEAD, "payload file bound")
                            saved_hash = file_sha256(path)
                            if idx == 107:
                                reference_files[hash_key] = (path, saved_hash)
                            manifest["files"].append(dict(path=filename, idx=idx, anchor=ANCHORS[a],
                                layer=layer, kind=name, component=component, shape=list(cpu.shape), dtype=str(cpu.dtype),
                                tensor_bytes=nbytes, file_bytes=path.stat().st_size,
                                tensor_sha256=tensor_hash,
                                file_sha256=saved_hash))
                            manifest["logical_tensor_bytes"] += nbytes
                            del cpu
            require(manifest["logical_tensor_bytes"] == PAYLOAD_LOGICAL_BYTES, "payload total logical bytes")
            require(len(manifest["files"]) == 6 * (2 * self.storage.layout.layers + 2 * self.storage.layout.kv_layers),
                    "payload tensor membership")
            manifest["status"] = "complete"
            path = directory / "manifest.json"
            encoded = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
            require(len(encoded.encode()) <= PAYLOAD_MANIFEST_LIMIT, "payload manifest bound")
            with path.open("x") as stream:
                stream.write(encoded)
            self.payload_status = dict(enabled=True, status="complete", directory=directory.name,
                manifest=path.name, manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                logical_tensor_bytes=manifest["logical_tensor_bytes"], tensor_files=len(manifest["files"]))
        except Exception as exc:
            self.payload_status = dict(enabled=True, status="failed", directory=directory.name)
            manifest.update(status="failed", error_type=type(exc).__name__)
            with (directory / "failed.json").open("x") as stream:
                json.dump(manifest, stream, indent=2, allow_nan=False)
                stream.write("\n")
            raise

    def snapshot(self):
        free, _ = torch.cuda.mem_get_info(self.runner.device)
        expected = {(idx, anchor) for idx in SELECTED for anchor in range(4)}
        return dict(performance_eligible=False, private_tensor_bytes=getattr(self, "private_bytes", PRIVATE_BYTES),
                    allocator_increment_bytes=self.allocator_increment,
                    incremental_guard_bytes=GUARD_BYTES, current_device_free_bytes=free,
                    memory_guard_pass=free >= GUARD_BYTES + (4 << 20),
                    anchors_complete=self.published == self.enqueued == expected,
                    pending_anchors=len(self.enqueued - self.published),
                    records=[self.records[k] for k in sorted(self.records)],
                    **({"payload_export": self.payload_status} if getattr(self, "payload_enabled", False) else {}),
                    scope="selected entry-state bytes only; dense accepted-prefix/rejected-suffix and carried-state correctness follow-up still required")
