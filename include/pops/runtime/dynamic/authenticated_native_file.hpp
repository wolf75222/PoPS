/// @file
/// @brief Platform load policy for a content-authenticated native artifact.

#pragma once

#include <pops/core/identity/sha256.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#include <climits>
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace pops::dynlib {

/// Owns one absolute, private copy of a self-contained native artifact. The token authenticates the
/// bytes written to this copy, and the backing file remains write/delete-excluded until its loaded
/// module has been unloaded. Generated packages may resolve host-global/system libraries, but
/// relative ``$ORIGIN``/``@loader_path``/DLL-adjacent dependencies are intentionally unsupported.
class AuthenticatedNativeFile final {
 public:
  explicit AuthenticatedNativeFile(const std::string& path) {
#if defined(_WIN32)
    const std::wstring source_path = utf8_to_wide(path);
    HANDLE source =
        ::CreateFileW(source_path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                      FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (source == INVALID_HANDLE_VALUE)
      throw std::runtime_error("cannot open native artifact '" + path + "'");
    try {
      if (::GetFileType(source) != FILE_TYPE_DISK)
        throw std::runtime_error("native artifact is not a regular disk file");
      LARGE_INTEGER size{};
      if (!::GetFileSizeEx(source, &size) || size.QuadPart < 0)
        throw std::runtime_error("cannot determine native artifact size");
      if (static_cast<std::uint64_t>(size.QuadPart) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        throw std::runtime_error("native artifact exceeds addressable size");

      std::array<wchar_t, MAX_PATH + 1> raw_tmp{};
      const DWORD tmp_size = ::GetTempPathW(static_cast<DWORD>(raw_tmp.size()), raw_tmp.data());
      if (tmp_size == 0 || tmp_size >= raw_tmp.size())
        throw std::runtime_error("cannot resolve an absolute native loader temp directory");
      std::array<wchar_t, 32768> absolute_tmp{};
      const DWORD absolute_size = ::GetFullPathNameW(
          raw_tmp.data(), static_cast<DWORD>(absolute_tmp.size()), absolute_tmp.data(), nullptr);
      if (absolute_size == 0 || absolute_size >= absolute_tmp.size() ||
          !(absolute_tmp[0] == L'\\' || (absolute_size >= 3 && absolute_tmp[1] == L':' &&
                                         (absolute_tmp[2] == L'\\' || absolute_tmp[2] == L'/'))))
        throw std::runtime_error("native loader temp directory is not absolute");
      std::wstring tmp_root(absolute_tmp.data(), absolute_size);
      if (!tmp_root.empty() && tmp_root.back() != L'\\' && tmp_root.back() != L'/')
        tmp_root.push_back(L'\\');
      for (int attempt = 0; attempt < 64 && shadow_directory_w_.empty(); ++attempt) {
        std::wstring candidate = tmp_root + L"pops-native-" + random_token();
        if (::CreateDirectoryW(candidate.c_str(), nullptr))
          shadow_directory_w_ = std::move(candidate);
        else if (::GetLastError() != ERROR_ALREADY_EXISTS)
          throw std::runtime_error("cannot create private native artifact directory");
      }
      if (shadow_directory_w_.empty())
        throw std::runtime_error("cannot allocate a unique native artifact directory");
      directory_handle_ =
          ::CreateFileW(shadow_directory_w_.c_str(), FILE_LIST_DIRECTORY | FILE_READ_ATTRIBUTES,
                        FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                        FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
      if (directory_handle_ == INVALID_HANDLE_VALUE)
        throw std::runtime_error("cannot pin private native artifact directory");
      for (int attempt = 0; attempt < 64 && handle_ == INVALID_HANDLE_VALUE; ++attempt) {
        load_path_w_ = shadow_directory_w_ + L"\\image-" + random_token() + L".dll";
        handle_ = ::CreateFileW(load_path_w_.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ,
                                nullptr, CREATE_NEW,
                                FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (handle_ == INVALID_HANDLE_VALUE && ::GetLastError() != ERROR_FILE_EXISTS &&
            ::GetLastError() != ERROR_ALREADY_EXISTS)
          break;
      }
      if (handle_ == INVALID_HANDLE_VALUE)
        throw std::runtime_error("cannot create private native artifact copy");

      std::array<std::uint8_t, 1024 * 1024> buffer{};
      std::vector<std::uint8_t> authenticated_bytes;
      authenticated_bytes.reserve(static_cast<std::size_t>(size.QuadPart));
      for (;;) {
        DWORD read = 0;
        if (!::ReadFile(source, buffer.data(), static_cast<DWORD>(buffer.size()), &read, nullptr))
          throw std::runtime_error("cannot read native artifact source");
        if (read == 0)
          break;
        authenticated_bytes.insert(authenticated_bytes.end(), buffer.begin(),
                                   buffer.begin() + read);
        DWORD offset = 0;
        while (offset < read) {
          DWORD written = 0;
          if (!::WriteFile(handle_, buffer.data() + offset, read - offset, &written, nullptr) ||
              written == 0)
            throw std::runtime_error("cannot write private native artifact copy");
          offset += written;
        }
      }
      if (authenticated_bytes.size() != static_cast<std::size_t>(size.QuadPart) ||
          !::FlushFileBuffers(handle_))
        throw std::runtime_error("private native artifact copy is incomplete");
      FILE_BASIC_INFO attributes{};
      if (!::GetFileInformationByHandleEx(handle_, FileBasicInfo, &attributes, sizeof(attributes)))
        throw std::runtime_error("cannot inspect private native artifact attributes");
      attributes.FileAttributes = FILE_ATTRIBUTE_READONLY;
      if (!::SetFileInformationByHandle(handle_, FileBasicInfo, &attributes, sizeof(attributes)))
        throw std::runtime_error("cannot seal private native artifact copy");
      LARGE_INTEGER copied_size{};
      if (!::GetFileSizeEx(handle_, &copied_size) || copied_size.QuadPart < 0 ||
          copied_size.QuadPart != static_cast<LONGLONG>(authenticated_bytes.size()))
        throw std::runtime_error("private native artifact size differs from copied bytes");
      LARGE_INTEGER beginning{};
      if (!::SetFilePointerEx(handle_, beginning, nullptr, FILE_BEGIN))
        throw std::runtime_error("cannot rewind sealed native artifact copy");
      std::vector<std::uint8_t> mapped_bytes(static_cast<std::size_t>(copied_size.QuadPart));
      std::size_t mapped_offset = 0;
      while (mapped_offset < mapped_bytes.size()) {
        const DWORD request = static_cast<DWORD>(
            std::min<std::size_t>(mapped_bytes.size() - mapped_offset,
                                  static_cast<std::size_t>(std::numeric_limits<DWORD>::max())));
        DWORD read = 0;
        if (!::ReadFile(handle_, mapped_bytes.data() + mapped_offset, request, &read, nullptr) ||
            read == 0)
          throw std::runtime_error("cannot authenticate sealed native artifact copy");
        mapped_offset += read;
      }
      size_ = copied_size.QuadPart;
      content_sha256_ = identity::sha256_hex(mapped_bytes);
      // Retain the exact CREATE_NEW file object continuously, but restrict the surviving handle to
      // read access before LoadLibraryExW opens its image section. DuplicateHandle does not resolve
      // the pathname again; the original sharing contract still excludes every competing writer or
      // deleter, so the mapped bytes cannot differ from those authenticated above.
      HANDLE read_only = INVALID_HANDLE_VALUE;
      if (!::DuplicateHandle(::GetCurrentProcess(), handle_, ::GetCurrentProcess(), &read_only,
                             GENERIC_READ, FALSE, 0))
        throw std::runtime_error("cannot retain sealed native artifact file authority");
      const HANDLE writable = handle_;
      handle_ = read_only;
      (void)::CloseHandle(writable);
      load_path_ = wide_to_utf8(load_path_w_);
    } catch (...) {
      if (handle_ != INVALID_HANDLE_VALUE) {
        ::CloseHandle(handle_);
        handle_ = INVALID_HANDLE_VALUE;
      }
      if (!load_path_w_.empty()) {
        (void)::SetFileAttributesW(load_path_w_.c_str(), FILE_ATTRIBUTE_NORMAL);
        (void)::DeleteFileW(load_path_w_.c_str());
      }
      if (directory_handle_ != INVALID_HANDLE_VALUE) {
        ::CloseHandle(directory_handle_);
        directory_handle_ = INVALID_HANDLE_VALUE;
      }
      if (!shadow_directory_w_.empty())
        (void)::RemoveDirectoryW(shadow_directory_w_.c_str());
      ::CloseHandle(source);
      throw;
    }
    ::CloseHandle(source);
#else
    const int source = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (source < 0)
      throw std::runtime_error("cannot open native artifact '" + path + "'");
    try {
      struct stat status{};
      if (::fstat(source, &status) != 0 || status.st_size < 0)
        throw std::runtime_error("cannot determine native artifact size");
      if (!S_ISREG(status.st_mode))
        throw std::runtime_error("native artifact is not a regular file");
      if (static_cast<std::uint64_t>(status.st_size) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        throw std::runtime_error("native artifact exceeds addressable size");

      const char* configured_tmp = std::getenv("TMPDIR");
      const std::string requested_tmp = configured_tmp == nullptr || *configured_tmp == '\0'
                                            ? std::string("/tmp")
                                            : std::string(configured_tmp);
      if (requested_tmp.empty() || requested_tmp.front() != '/')
        throw std::runtime_error("native loader temp directory must be absolute");
      std::array<char, PATH_MAX> resolved_tmp{};
      if (::realpath(requested_tmp.c_str(), resolved_tmp.data()) == nullptr ||
          resolved_tmp.front() != '/')
        throw std::runtime_error("cannot resolve native loader temp directory");
      const std::string tmp_root = resolved_tmp.data();
      struct stat tmp_status{};
      if (::stat(tmp_root.c_str(), &tmp_status) != 0 || !S_ISDIR(tmp_status.st_mode) ||
          ::access(tmp_root.c_str(), W_OK | X_OK) != 0)
        throw std::runtime_error("native loader temp directory is not usable");
      const bool owned_root = tmp_status.st_uid == ::geteuid();
      const bool privileged_sticky_root =
          tmp_status.st_uid == 0 && (tmp_status.st_mode & S_ISVTX) != 0;
      if (!owned_root && !privileged_sticky_root)
        throw std::runtime_error("native loader temp directory has untrusted ownership");
      std::string directory_template = tmp_root + "/pops-native-XXXXXX";
      std::vector<char> writable(directory_template.begin(), directory_template.end());
      writable.push_back('\0');
      const char* directory = ::mkdtemp(writable.data());
      if (directory == nullptr)
        throw std::runtime_error("cannot create private native artifact directory");
      shadow_directory_ = directory;
      struct stat private_status{};
      if (::stat(shadow_directory_.c_str(), &private_status) != 0 ||
          private_status.st_uid != ::geteuid() || (private_status.st_mode & 077) != 0)
        throw std::runtime_error("native artifact directory is not private to the current user");
#if defined(__APPLE__)
      constexpr std::string_view image_suffix = ".dylib";
#else
      constexpr std::string_view image_suffix = ".so";
#endif
      std::string image_template = shadow_directory_ + "/image-XXXXXX" + std::string(image_suffix);
      std::vector<char> image_writable(image_template.begin(), image_template.end());
      image_writable.push_back('\0');
      fd_ = ::mkstemps(image_writable.data(), static_cast<int>(image_suffix.size()));
      if (fd_ < 0)
        throw std::runtime_error("cannot create private native artifact shadow");
      load_path_ = image_writable.data();
      const int descriptor_flags = ::fcntl(fd_, F_GETFD);
      if (descriptor_flags < 0 || ::fcntl(fd_, F_SETFD, descriptor_flags | FD_CLOEXEC) != 0)
        throw std::runtime_error("cannot protect private native artifact descriptor");
      std::array<std::uint8_t, 1024 * 1024> buffer{};
      std::vector<std::uint8_t> authenticated_bytes;
      authenticated_bytes.reserve(static_cast<std::size_t>(status.st_size));
      for (;;) {
        ssize_t count = ::read(source, buffer.data(), buffer.size());
        if (count < 0 && errno == EINTR)
          continue;
        if (count < 0)
          throw std::runtime_error("cannot read native artifact source");
        if (count == 0)
          break;
        authenticated_bytes.insert(authenticated_bytes.end(), buffer.begin(),
                                   buffer.begin() + count);
        std::size_t offset = 0;
        while (offset < static_cast<std::size_t>(count)) {
          const ssize_t written =
              ::write(fd_, buffer.data() + offset, static_cast<std::size_t>(count) - offset);
          if (written < 0 && errno == EINTR)
            continue;
          if (written <= 0)
            throw std::runtime_error("cannot write private native artifact shadow");
          offset += static_cast<std::size_t>(written);
        }
      }
      if (::fsync(fd_) != 0 || ::fchmod(fd_, S_IRUSR | S_IXUSR) != 0)
        throw std::runtime_error("cannot seal private native artifact shadow");
      struct stat shadow_status{};
      if (::fstat(fd_, &shadow_status) != 0 || shadow_status.st_size < 0)
        throw std::runtime_error("cannot determine private native artifact size");
      size_ = static_cast<std::int64_t>(shadow_status.st_size);
      if (size_ != static_cast<std::int64_t>(authenticated_bytes.size()))
        throw std::runtime_error("private native artifact shadow is incomplete");
      std::vector<std::uint8_t> mapped_bytes(static_cast<std::size_t>(size_));
      std::size_t mapped_offset = 0;
      while (mapped_offset < mapped_bytes.size()) {
        const ssize_t count =
            ::pread(fd_, mapped_bytes.data() + mapped_offset, mapped_bytes.size() - mapped_offset,
                    static_cast<off_t>(mapped_offset));
        if (count < 0 && errno == EINTR)
          continue;
        if (count <= 0)
          throw std::runtime_error("cannot authenticate private native artifact shadow");
        mapped_offset += static_cast<std::size_t>(count);
      }
      content_sha256_ = identity::sha256_hex(mapped_bytes);
    } catch (...) {
      if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
      }
      if (!load_path_.empty())
        (void)::unlink(load_path_.c_str());
      if (!shadow_directory_.empty())
        (void)::rmdir(shadow_directory_.c_str());
      ::close(source);
      throw;
    }
    ::close(source);
#endif
  }

  AuthenticatedNativeFile(const AuthenticatedNativeFile&) = delete;
  AuthenticatedNativeFile& operator=(const AuthenticatedNativeFile&) = delete;
  AuthenticatedNativeFile(AuthenticatedNativeFile&&) = delete;
  AuthenticatedNativeFile& operator=(AuthenticatedNativeFile&&) = delete;
  ~AuthenticatedNativeFile() {
#if defined(_WIN32)
    if (handle_ != INVALID_HANDLE_VALUE)
      ::CloseHandle(handle_);
    if (!load_path_w_.empty()) {
      (void)::SetFileAttributesW(load_path_w_.c_str(), FILE_ATTRIBUTE_NORMAL);
      (void)::DeleteFileW(load_path_w_.c_str());
    }
    if (directory_handle_ != INVALID_HANDLE_VALUE)
      ::CloseHandle(directory_handle_);
    if (!shadow_directory_w_.empty())
      (void)::RemoveDirectoryW(shadow_directory_w_.c_str());
#else
    if (fd_ >= 0) {
      ::close(fd_);
      (void)::unlink(load_path_.c_str());
      (void)::rmdir(shadow_directory_.c_str());
    }
#endif
  }

  [[nodiscard]] const std::string& content_sha256() const noexcept { return content_sha256_; }
  [[nodiscard]] std::int64_t size() const noexcept { return size_; }
  [[nodiscard]] std::string binary_identity() const {
    return identity::binary_identity_token(identity::sha256_hex_bytes(content_sha256_), size_);
  }
  [[nodiscard]] const std::string& load_path() const noexcept { return load_path_; }

 private:
#if defined(_WIN32)
  static std::wstring utf8_to_wide(const std::string& value) {
    const int count =
        ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.c_str(), -1, nullptr, 0);
    if (count <= 0)
      throw std::runtime_error("native artifact path is not valid UTF-8");
    std::wstring wide(static_cast<std::size_t>(count), L'\0');
    if (::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.c_str(), -1, wide.data(),
                              count) == 0)
      throw std::runtime_error("native artifact path conversion failed");
    wide.pop_back();
    return wide;
  }

  static std::string wide_to_utf8(const std::wstring& value) {
    const int count =
        ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.c_str(),
                              static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
    if (count <= 0)
      throw std::runtime_error("private native artifact path conversion failed");
    std::string utf8(static_cast<std::size_t>(count), '\0');
    if (::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.c_str(),
                              static_cast<int>(value.size()), utf8.data(), count, nullptr,
                              nullptr) == 0)
      throw std::runtime_error("private native artifact path conversion failed");
    return utf8;
  }

  static std::wstring random_token() {
    std::random_device source;
    constexpr wchar_t digits[] = L"0123456789abcdef";
    std::wstring token(32, L'0');
    for (wchar_t& value : token)
      value = digits[source() & 0x0fU];
    return token;
  }
#endif

  std::string content_sha256_;
  std::string load_path_;
  std::int64_t size_ = 0;
#if defined(_WIN32)
  HANDLE handle_ = INVALID_HANDLE_VALUE;
  HANDLE directory_handle_ = INVALID_HANDLE_VALUE;
  std::wstring load_path_w_;
  std::wstring shadow_directory_w_;
#else
  int fd_ = -1;
  std::string shadow_directory_;
#endif
};

}  // namespace pops::dynlib
