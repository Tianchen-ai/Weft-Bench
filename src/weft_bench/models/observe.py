"""Optional instrumentation of the llama-bench harness, outside its timed region."""

from pathlib import Path


def instrument(source, *, weft_include=None):
    root = Path(source)
    path = root / "tools/llama-bench/llama-bench.cpp"
    original = text = path.read_text()
    if "PROFILE_WINDOW " not in text:
        begin = "            uint64_t t_start = get_time_ns();"
        end = "            uint64_t t_ns = get_time_ns() - t_start;"
        if text.count(begin) != 1 or text.count(end) != 1:
            raise ValueError(
                "llama-bench timing boundary changed; update the observer anchors"
            )
        text = "#include <time.h>\n" + text
        text = text.replace(
            begin,
            "            struct timespec wb_begin, wb_end;\n"
            "            clock_gettime(CLOCK_MONOTONIC, &wb_begin);\n" + begin,
        )
        text = text.replace(
            end,
            end + """
            clock_gettime(CLOCK_MONOTONIC, &wb_end);
            fprintf(stderr, "PROFILE_WINDOW %ld.%09ld %ld.%09ld pp=%d tg=%d sample=%d\\n",
                    wb_begin.tv_sec, wb_begin.tv_nsec, wb_end.tv_sec, wb_end.tv_nsec,
                    t.n_prompt, t.n_gen, i);""",
        )
    if weft_include and "WEFT_CALLS type=" not in text:
        header = Path(weft_include) / "weft/ggml.h"
        if not header.is_file():
            raise ValueError(f"missing integration header {header}")
        anchor = "            t.samples_ns.push_back(t_ns);"
        if text.count(anchor) != 1:
            raise ValueError("llama-bench sample boundary changed")
        text = '#include "' + str(header.resolve()) + '"\n' + text
        text = text.replace(
            anchor,
            """            for (int type = 0; type < GGML_TYPE_COUNT; ++type) {
                ggml_weft_counters counts{};
                if (ggml_weft_counters_get(static_cast<ggml_type>(type), &counts) &&
                    (counts.vec_dot_calls || counts.weft_quantize_calls))
                    fprintf(stderr, "WEFT_CALLS type=%s dots=%llu quantizes=%llu\\n", ggml_type_name(static_cast<ggml_type>(type)),
                            (unsigned long long)counts.vec_dot_calls, (unsigned long long)counts.weft_quantize_calls);
            }
            fprintf(stderr, "WEFT_IME_CALLS %llu\\n", (unsigned long long)ggml_weft_ime1_calls());
""" + anchor,
        )
    if weft_include and "WB_COUNTER sample=" not in text:
        header = Path(weft_include) / "weft/ggml.h"
        helper = """
static void wb_measurement_counts(const char *point, int sample) {
  for(int type=0;type<GGML_TYPE_COUNT;++type) {
    ggml_weft_counters counts{};
    if(ggml_weft_counters_get(static_cast<ggml_type>(type), &counts))
      fprintf(stderr,"WB_COUNTER sample=%d point=%s type=%s dots=%llu quantizes=%llu ime=0\\n",sample,point,
              ggml_type_name(static_cast<ggml_type>(type)),(unsigned long long)counts.vec_dot_calls,(unsigned long long)counts.weft_quantize_calls);
  }
  fprintf(stderr,"WB_COUNTER sample=%d point=%s type=ime dots=0 quantizes=0 ime=%llu\\n",sample,point,(unsigned long long)ggml_weft_ime1_calls());
}
"""
        text = (
            "#include <cstdio>\n#include "
            + repr(str(header.resolve())).replace("'", '"')
            + "\n"
            + helper
            + text
        )
        starts = [
            "            struct timespec wb_begin, wb_end;",
            "            struct timespec profile_begin, profile_end;",
        ]
        start = next((s for s in starts if s in text), None)
        if not start:
            raise ValueError("missing timing-window declaration")
        text = text.replace(
            start, '            wb_measurement_counts("begin", i);\n' + start
        )
        anchor = "            t.samples_ns.push_back(t_ns);"
        text = text.replace(
            anchor, '            wb_measurement_counts("end", i);\n' + anchor
        )
    if text != original:
        path.write_text(text)
    return {
        "source": str(path),
        "changed": text != original,
        "counters": "WEFT_CALLS type=" in text,
    }


if __name__ == "__main__":
    import json
    import sys

    request = json.loads(Path(sys.argv[1]).read_text())
    print(
        json.dumps(
            instrument(request["source"], weft_include=request.get("weft_include"))
        )
    )
