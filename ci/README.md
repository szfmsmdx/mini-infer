# PR 检查

GitHub 在 PR 打开或更新时运行 `C++ correctness`。分支保护沿用已有的必需检查 `cpp-correctness`。新 push 取消同一 PR 的旧运行。

| 顺序 | 检查 | 通过条件 |
| --- | --- | --- |
| 1 | lint、编译 | Clang 18 检查改动行的格式和 CPU C++ 静态诊断。CPU 库、示例、测试全部编译成功。CUDA 源码参加格式检查。 |
| 2 | 行覆盖率 | GCC Debug 插桩，执行原生 CPU CTest，再由 gcovr 生成 XML 和文本报告。新增或修改的可执行行至少 70%，全量可测行至少 60%。 |
| 3 | 正确性 | 对 GitHub 生成的候选合并提交重新构建 Release，运行完整 CPU CTest，包含 PyTorch 精度对照。失败或跳过都不能通过。 |

**格式来源。** [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/.clang-format) 同样用四空格、同一行左花括号、namespace 顶格。mini-infer 保留 `Tensor&` 写法和项目头文件的依赖顺序，formatter 不重排 include。[PyTorch](https://github.com/pytorch/pytorch/blob/main/.clang-tidy) 使用 bugprone、performance 和重复 include 检查。这里先选 sizeof、memset、移动后使用等具体规则，不强制改名或使用 C++20。旧代码随所在行的改动逐步检查。

**覆盖范围。** 分母取自 `src/`、`include/`、`examples/` 的 CPU 编译报告，同一文件同一行只计一次；测试、第三方库和构建产物排除。增量使用 PR head 相对目标分支共同祖先的新增行，与该 head 的可执行行求交。删除行不计。纯文档 PR 的增量为 N/A，全量仍须达标。报告可在 run 的 Summary 和 `cpu-line-coverage` artifact 查看。

GitHub 托管的 Ubuntu runner 无 GPU，CPU 行覆盖率不能代表 CUDA 设备代码。变更生产 `.cu/.cuh` 或缺少该文件的 CPU 测量时要求特批，请在 PR 附上 GPU 验证结果。基础设施错误、覆盖率采集测试失败、空报告直接阻断流程。

**特批按钮。** 低于门槛时，打开 PR → Checks → 当前 Actions run → **Review deployments**，选 `coverage-exception` 并批准或拒绝。[GitHub 审批说明](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/review-deployments)。环境的唯一 required reviewer 是仓库所有者，允许审自己的 PR，禁用管理员直接跳过。批准仅对当前运行生效，之后仍须通过完整正确性测试。

**审查状态。** GitHub 的 Draft 已是 open PR。默认分支上的 `Mark tested draft ready` 只将 CI 成功、仍指向同一提交的 Draft 转为 Ready for review。它不 checkout PR 代码。首次引入这个工作流的 PR 需要在验证通过后手动转 Ready；合并决定由用户作出。
