"""Selected native graph activations: byte counts only, never model feedback."""

import math
from dataclasses import dataclass

import torch

PRIVATE_LIMIT = 4 << 20
GUARD = 512 << 20
_observer = None


def require(value, message):
    if not value:
        raise RuntimeError("B4 layer-boundary diagnostic: " + message)


@dataclass(frozen=True)
class Layout:
    layers: int = 40
    hidden: int = 2048
    steps: int = 5

    @property
    def sites(self):
        return self.layers * 4 + 1

    def specs(self):
        n, d, h = self.sites, self.steps, self.hidden
        return dict(reference=((n, d, h), torch.bfloat16),
                    presence=((n,), torch.bool), generation=((n,), torch.int64),
                    counts=((n, d), torch.int64), flags=((n, 4), torch.int64),
                    visits=((2, n), torch.int64), control=((4,), torch.int64),
                    rank=((d,), torch.int64), indices=((d,), torch.int64),
                    work=((d, h), torch.bfloat16), different=((d, h * 2), torch.bool),
                    count_work=((d,), torch.int64), present_work=((), torch.bool),
                    negative=((), torch.int64))

    def nbytes(self):
        return sum(math.prod(shape) * dtype.itemsize for shape, dtype in self.specs().values())


class BoundaryBuffers:
    """Captured operations read device controls at replay, not Python row state.

    One immutable reference bank; scratch is used only on the native model
    stream. Scalars in control are enable, reference, selected row, generation.
    """

    def __init__(self, device, layout=Layout()):
        self.layout = layout
        self.buffers = {name: torch.zeros(shape, dtype=dtype, device=device)
                        for name, (shape, dtype) in layout.specs().items()}
        self.buffers["rank"].copy_(torch.arange(layout.steps, device=device))
        self.buffers["counts"].fill_(-1)
        self.buffers["negative"].fill_(-1)

    def disable(self):
        self.buffers["control"][0].zero_()

    def activate(self, reference, row, generation):
        for i, value in enumerate((1, int(reference), row, generation)):
            self.buffers["control"][i].fill_(value)

    def tap(self, site, tensor, rows):
        s, b = self.layout, self.buffers
        require(0 <= site < s.sites and rows > 0, "site/batch shape")
        present = tensor is not None
        if present:
            require(tensor.ndim == 2 and tensor.shape == (rows * s.steps, s.hidden)
                    and tensor.dtype == torch.bfloat16, "native activation shape/dtype")
        enabled, reference, row, generation = b["control"].unbind()
        active = enabled == 1
        ref, compare = active & (reference == 1), active & (reference == 0)
        # Safe inert replay even on smaller graph buckets; active rows are
        # independently bound to the actual graph by the outside-graph hook.
        torch.add(b["rank"], row.clamp(0, rows - 1) * s.steps, out=b["indices"])
        if present:
            torch.index_select(tensor, 0, b["indices"], out=b["work"])
        else:
            b["work"].zero_()
        old = b["generation"][site]
        b["flags"][site, 0].bitwise_or_((compare & (old != 1)).long())
        b["flags"][site, 1].bitwise_or_((active & ((generation != 2 - reference)
                                                | ((reference != 0) & (reference != 1)))).long())
        b["flags"][site, 2].bitwise_or_((compare & (b["presence"][site] != present)).long())
        b["flags"][site, 3].bitwise_or_(((ref & (old != 0))
                                     | (active & ((row < 0) | (row >= rows)))).long())
        write = ref & (old == 0)
        torch.where(write, b["work"], b["reference"][site], out=b["reference"][site])
        b["present_work"].fill_(present)
        torch.where(write, b["present_work"], b["presence"][site], out=b["presence"][site])
        torch.where(write, generation, old, out=old)
        torch.ne(b["work"].view(torch.uint8), b["reference"][site].view(torch.uint8), out=b["different"])
        torch.sum(b["different"], dim=1, dtype=torch.int64, out=b["count_work"])
        torch.where(reference == 1, b["negative"], b["count_work"], out=b["count_work"])
        torch.where(active, b["count_work"], b["counts"][site], out=b["counts"][site])
        b["visits"][0, site].add_(ref.long())
        b["visits"][1, site].add_(compare.long())

    def owned_result(self):
        return {"boundary_" + name: self.buffers[name].clone()
                for name in ("counts", "flags", "visits", "presence", "generation", "control")}


class LayerObserver:
    def __init__(self, runner, observer):
        expected = dict(client=1, indices=[107, 187], anchor=9, layers=40, hidden=2048,
                        steps=5, private_limit_bytes=PRIVATE_LIMIT, incremental_guard_bytes=GUARD)
        require(observer.entry is not None and observer.config.get("layer_boundary") == expected,
                "exact opt-in requires existing matched entry observation")
        config = runner.model_config.hf_text_config
        require(config.num_hidden_layers == 40 and config.hidden_size == 2048
                and config.model_type == "qwen3_5_moe_text", "native target model")
        self.runner, self.observer = runner, observer
        self.layout = Layout()
        needed = self.layout.nbytes()
        require(needed <= PRIVATE_LIMIT, "private tensor cap")
        free, _ = torch.cuda.mem_get_info(runner.device)
        require(free >= needed + GUARD + (4 << 20), "incremental allocation guard")
        before = torch.cuda.memory_allocated(runner.device)
        self.storage = BoundaryBuffers(runner.device, self.layout)
        self.allocator_increment = torch.cuda.memory_allocated(runner.device) - before
        require(torch.cuda.mem_get_info(runner.device)[0] >= GUARD + (4 << 20),
                "postallocation physical headroom")
        self.enqueued, self.records = set(), {}
        self.stream = None

    def before_verify(self):
        self.storage.disable()

    def graph_entry(self, graph, forward):
        selected = getattr(forward, "_b4_entry_selection", None)
        if selected is None or selected[3] != 4:
            return
        row, req, idx, _ = selected
        require(idx in (107, 187) and idx not in self.enqueued, "duplicate selected anchor")
        require(idx == 107 or 107 in self.enqueued, "comparison before reference")
        require(graph.model_runner is self.runner and graph.bs == graph.raw_bs == 32
                and row == 31 and forward.batch_size == 32, "native matching graph/row")
        stream = torch.cuda.current_stream(self.runner.device).cuda_stream
        if self.stream is None:
            self.stream = stream
        require(stream == self.stream and not torch.cuda.is_current_stream_capturing(),
                "native producing stream changed")
        self.storage.activate(idx == 107, row, 1 if idx == 107 else 2)
        forward._b4_boundary_selection = dict(rid=req.rid, idx=idx, anchor=9, row=row,
                                             generation=1 if idx == 107 else 2)
        self.enqueued.add(idx)

    def take(self, forward):
        meta = getattr(forward, "_b4_boundary_selection", None)
        if meta is None:
            return {}, None
        del forward._b4_boundary_selection
        return self.storage.owned_result(), dict(meta, sites=self.layout.sites,
            fields=["input_hidden", "input_residual", "output_hidden", "output_residual"],
            final_site="final_normalized_hidden", private_tensor_bytes=self.layout.nbytes(),
            private_limit_bytes=PRIVATE_LIMIT, incremental_guard_bytes=GUARD)

    def publish(self, req, record):
        meta = record["metadata"].get("layer_boundary")
        if meta is None or meta["rid"] != req.rid:
            return
        idx, data = meta["idx"], record["tensors"]
        require(idx in self.enqueued and idx not in self.records, "publication generation")
        require(record["start_position"] == 9 and record["repetition"] == 1,
                "anchor publication identity")
        entry = record.get("entry_state")
        require(entry is not None and entry["anchor"] == 9 and entry["idx"] == idx
                and entry["byte_differences"] == ([-1] * 80 if idx == 107 else [0] * 80),
                "boundary comparison requires matched selected entry bytes")
        validate_result(data, idx == 107, self.layout)
        counts = data["boundary_counts"]
        first = next((i for i, values in enumerate(counts) if max(values) > 0), None)
        summary = dict(meta, mismatch_bytes_by_site_token=counts,
                       first_differing_site=first, snapshot=record["snapshot"],
                       first_differing_boundary=(None if first is None else
                           dict(layer=first // 4, field=meta["fields"][first % 4])
                           if first < self.layout.layers * 4 else dict(layer=None, field="final_normalized_hidden")),
                       scope="Selected five-token decoder boundaries only; no operator/cause or all-batch claim")
        record["layer_boundary"] = summary
        self.records[idx] = summary

    def snapshot(self):
        free, _ = torch.cuda.mem_get_info(self.runner.device)
        require(free >= GUARD + (4 << 20), "physical headroom")
        return dict(performance_eligible=False, private_tensor_bytes=self.layout.nbytes(),
                    private_limit_bytes=PRIVATE_LIMIT, allocator_increment_bytes=self.allocator_increment,
                    incremental_guard_bytes=GUARD, current_device_free_bytes=free,
                    pending_anchors=len(self.enqueued - self.records.keys()),
                    complete=set(self.records) == {107, 187}, records=list(self.records.values()))


def validate_result(data, reference, layout=Layout()):
    n, d = layout.sites, layout.steps
    require(data["boundary_flags"] == [[0] * 4 for _ in range(n)], "device ownership/epoch failure")
    require(data["boundary_visits"] == [[1] * n, [0 if reference else 1] * n], "missing/duplicate graph tap")
    require(data["boundary_generation"] == [1] * n, "reference overwritten or uninitialized")
    require(data["boundary_control"] == [1, int(reference), 31, 1 if reference else 2],
            "selected device control identity")
    require(data["boundary_presence"] == [i != 1 for i in range(n)], "native residual presence")
    counts = data["boundary_counts"]
    require(len(counts) == n and all(len(row) == d and all(type(x) is int and
            (x == -1 if reference else 0 <= x <= layout.hidden * 2) for x in row) for row in counts),
            "byte-count bound")


def initialize(runner, observer):
    global _observer
    require(_observer is None, "duplicate initialization")
    _observer = LayerObserver(runner, observer)
    return _observer


def boundary(forward, layer, edge, hidden, residual):
    if _observer is None or not forward.forward_mode.is_target_verify():
        return
    require(edge in ("input", "output") and 0 <= layer < 40, "layer boundary")
    site = layer * 4 + (0 if edge == "input" else 2)
    _observer.storage.tap(site, hidden, hidden.shape[0] // 5)
    _observer.storage.tap(site + 1, residual, hidden.shape[0] // 5)


def final_hidden(forward, hidden):
    if _observer is not None and forward.forward_mode.is_target_verify():
        _observer.storage.tap(160, hidden, hidden.shape[0] // 5)
