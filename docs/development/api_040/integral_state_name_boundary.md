# Integral identity: embedded NUL refusal

A host reproduction against `88eba605` used the actual `emit_integral_declarations`
for the author name `q\0tail`. Clang C++20 accepted the generated `\u0000`
inside the string literal. The conversion to the native `const std::string&`
parameter truncated the identifier: Python expected 112 UTF-8 bytes, while the
native declaration received 107. The host witness exited 1. This was identity
loss, not a compiler refusal in this environment.

The authoring boundary now rejects an embedded NUL before inserting the scalar
into the Program. Empty names and `/` retain their previous refusal. No additional
Unicode restriction is introduced. An independent host test compiles the real
emitter for ASCII, accented Latin, Greek and CJK names, prints the received bytes,
and compares them exactly with the Python identities. This is a host declaration
boundary check, not an installed-runtime receipt.
